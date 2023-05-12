# Use transformer to translate Portuguese to English
# env: tensorflow-gpu==2.12.0 tensorflow-text==2.12.1 tensorflow-datasets==4.9.2
# ref: https://www.tensorflow.org/text/tutorials/transformer

import logging
import time

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROP_RATE = 0.1


def prepare_data_set():
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    def prepare_batch(pt, en):
        pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
        pt = pt[:, :MAX_TOKENS]  # Trim to MAX_TOKENS.
        pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

        en = tokenizers.en.tokenize(en)
        en = en[:, :(MAX_TOKENS + 1)]
        en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
        en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

        return (pt, en_inputs), en_labels

    def make_batches(ds):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

    return tokenizers, make_batches(train_examples), make_batches(val_examples)


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


def test_positional_encoding():
    pos_encoding = positional_encoding(length=2048, depth=512)

    # Check the shape.
    print(pos_encoding.shape)

    # Plot the dimensions.
    plt.pcolormesh(pos_encoding.numpy().T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

    pos_encoding /= tf.norm(pos_encoding, axis=1, keepdims=True)
    p = pos_encoding[1000]
    dots = tf.einsum('pd,d -> p', pos_encoding, p)
    plt.subplot(2, 1, 1)
    plt.plot(dots)
    plt.ylim([0, 1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
             [0, 1, float('nan'), 0, 1], color='k', label='Zoom')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(dots)
    plt.xlim([950, 1050])
    plt.ylim([0, 1])
    plt.show()


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]  # (batch, seq)
        x = self.embedding(x)  # (batch, seq, d_model)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # (batch, seq, d_model)
        x = x + self.pos_encoding[tf.newaxis, :length, :]  # (batch, seq, d_model)
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        # (batch, seq, attn_dim), (batch, num_heads, target_seq, input_seq)
        attn_output, attn_scores = self.mha(x, key=context, value=context,
                                            return_attention_scores=True)
        print('CrossAttention attn_output shape: {}, attn_scores shape: {}'.format(attn_output.shape,
                                                                                   attn_scores.shape))

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores  # (batch, num_heads, target_seq, input_seq)

        x = self.add([x, attn_output])  # (batch, seq, attn_dim)
        x = self.layernorm(x)  # (batch, seq, attn_dim)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        # (batch, seq, attn_dim), (batch, num_heads, seq, seq)
        attn_output, attn_scores = self.mha(query=x, value=x, key=x,
                                            return_attention_scores=True)
        print('GlobalSelfAttention attn_output shape: {}, attn_scores shape: {}'.format(attn_output.shape,
                                                                                        attn_scores.shape))
        x = self.add([x, attn_output])  # (batch, seq, attn_dim)
        x = self.layernorm(x)  # (batch, seq, attn_dim)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add([x, attn_output])  # (batch, seq, attn_dim)
        x = self.layernorm(x)  # (batch, seq, attn_dim)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])  # (batch, seq, d_model)
        x = self.layernorm(x)  # (batch, seq, d_model)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = GlobalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x):
        x = self.self_attn(x)  # (batch, seq, d_model)
        x = self.ffn(x)  # (batch, seq, d_model)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = CausalSelfAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.cross_attn = CrossAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)
        self.ffn = FeedForward(d_model=d_model, dff=dff, dropout_rate=dropout_rate)

    def call(self, x, context):
        x = self.self_attn(x)  # (batch, seq, d_model)
        x = self.cross_attn(x, context)  # (batch, seq, d_model)

        # Cache the last attention scores for plotting later.
        self.last_cross_attn_scores = self.cross_attn.last_attn_scores  # (batch, num_heads, target_seq, input_seq)

        x = self.ffn(x)  # (batch, seq, d_model)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)  # (batch, seq, d_model)

        x = self.dropout(x)  # (batch, seq, d_model)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)  # (batch, seq, d_model)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)  # (batch, seq, d_model)

        x = self.dropout(x)  # (batch, seq, d_model)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)  # (batch, seq, d_model)

        self.last_attn_scores = self.dec_layers[-1].last_cross_attn_scores

        return x


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size, dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        print('context shape: {}, x shape: {}'.format(context.shape, x.shape))

        context = self.encoder(context)  # (batch, seq, d_model)
        print('encoder output shape: {}'.format(context.shape))

        x = self.decoder(x, context)  # (batch, seq, d_model)
        print('decoder output shape: {}'.format(x.shape))

        # Final linear layer
        logits = self.final_layer(x)  # (batch, seq, target_vocab_size)
        print('logits shape: {}'.format(logits.shape))

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            'd_model': self.d_model.numpy(),
            'warmup_steps': self.warmup_steps
        }


def masked_loss(real, pred):
    mask = real != 0  # example padded with 0
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                             reduction='none')  # reduce loss later with mask
    loss = loss_obj(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)  # average loss
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


def train(transformer, train_examples, val_examples):
    learning_rate = CustomSchedule(d_model=D_MODEL)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
    transformer.summary()
    ckpt_dir = 'checkpoints/transformer'
    ckpt_path = os.path.join(ckpt_dir, 'ckpt')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                           save_weights_only=True),  # checkpoint
        tf.keras.callbacks.TensorBoard(log_dir='logs/transformer')  # tensorboard
    ]
    # load saved model ckpt
    if os.path.exists(ckpt_dir):
        print('load saved model ckpt')
        transformer.load_weights(ckpt_path)

    transformer.fit(train_examples, epochs=20, validation_data=val_examples, callbacks=callbacks)


class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]  # (1, seq)

        sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()  # (1, seq)
        encoder_input = sentence  # (1, seq)

        # Get [START] [END] tokenize index.
        start_end = self.tokenizers.en.tokenize([''])[0]  # (2)
        start = start_end[0][tf.newaxis]  # (1)
        end = start_end[1][tf.newaxis]  # (1)

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)  # add [START] token

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())  # (1, i)
            predictions = self.transformer((encoder_input, output), training=False)  # (1, i, vocab_size)

            # select the last token from the seq_len dimension
            prediction_id = predictions[:, -1:, :]  # (1, 1, vocab_size)
            # get the index of the predicted token
            prediction_id = tf.cast(tf.argmax(prediction_id, axis=-1), tf.int64)  # (1, 1)
            # add new token to the output for next iteration
            output_array = output_array.write(i + 1, prediction_id[0])

            # check if the predicted token is the end token
            if prediction_id == end:
                break

        output = tf.transpose(output_array.stack())  # (1, seq_len)
        # decode the predicted ids to get the translated sentence
        text = self.tokenizers.en.detokenize(output)[0]  # (seq_len)
        # lookup token index to text
        tokens = tokenizers.en.lookup(output)[0]  # (seq_len)

        # get decoder last attention weights
        self.transformer((encoder_input, output[:, :-1]), training=False)
        attention_weights = self.transformer.decoder.last_attn_scores  # (1, num_heads, target_seq, input_seq)
        print('attention_weights shape: {}'.format(attention_weights.shape))

        return text, tokens, attention_weights


def translate(translator, sentence):
    text, tokens, attention_weights = translator(sentence)
    print('sentence: {}'.format(sentence))
    print('text: {}'.format(text.numpy().decode('utf-8')))
    print('tokens: {}'.format(tokens.numpy()))
    return text, tokens, attention_weights


class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (text, tokens, attention_weights) = self.translator(sentence)
        return text


if __name__ == '__main__':
    # prepare data set
    tokenizers, train_examples, val_examples = prepare_data_set()

    # test_positional_encoding()

    # build translator transformer model
    transformer = Transformer(num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF,
                              input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                              target_vocab_size=tokenizers.en.get_vocab_size().numpy(), dropout_rate=DROP_RATE)

    for (pt, en_inputs), en_labels in train_examples.take(1):
        output = transformer((pt, en_inputs))
        print('pt shape: {}, en_inputs shape: {}, en_labels shape: {}, output shape: {}'
              .format(pt.shape, en_inputs.shape, en_labels.shape, output.shape))

    # train translator
    train(transformer, train_examples, val_examples)

    # test translator
    translator = Translator(tokenizers, transformer)
    translate(translator, tf.constant(['este é um problema que temos que resolver.']))

    # export translator model
    export_translator = ExportTranslator(translator)
    tf.saved_model.save(export_translator, export_dir='translator')

    # load saved translator model
    loaded_translator = tf.saved_model.load('translator')
    port = 'este é um problema que temos que resolver.'
    eng = loaded_translator(port).numpy()
    print('portuguese: {}, english: {}'.format(port, eng.decode('utf-8')))
