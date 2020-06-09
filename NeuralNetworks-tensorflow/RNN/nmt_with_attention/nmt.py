#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Build neural machine translation model with attention. Use it to train a
model to translation english and chinese.
Run in tensorflow 2.1.0

Authors: zhaochaochao(zhaochaochao@baidu.com)
Date:    2020/6/6 下午5:03
"""

# common libs.
import os
import time
import re
import io
import unicodedata

# 3rd-part libs.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

print(tf.__version__)

# Download dataset of chinese-english. Download from
# http://www.manythings.org/anki/ directly because can't access google storage.
# path_to_zip = tf.keras.utils.get_file(
#     'cmn-eng.zip',
#     origin='http://storage.googleapis.com/download.tensorflow.org/data/cmn-eng'
#            '.zip',
#     extract=True)
PATH_TO_FILE = '/home/zhaocc/.keras/datasets/cmn-eng/cmn.txt'
BATCH_SIZE = 64
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 50
MODEL_PATH = '/tmp/nmt'
has_show_summary = False  # Show summary only once.


def unicode_to_ascii(s):
    """Converts from unicode file to ascii."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if
                   unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    """Preprocess sentence."""
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # creating a space between a chinese word.
    # eg: "他是一个男孩。" => "他 是 一 个 男 孩 。"
    w = re.sub(r"([?.!,？。！，、\u4e00-\u9fa5])", r" \1 ", w)
    w = re.sub(r'[" "“”]+', " ", w)  # replace quotation marks to space.

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",",
    # "。", "？", "！", "，", "、", "所有汉字")
    w = re.sub(r"[^a-zA-Z\u4e00-\u9fa5?.!,？。！，、]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


def create_dataset(path, num_examples=None):
    """Create dataset.

    Args:
        path: The file path.
        num_examples: The total examples number.

    Returns:
        word pairs in the format: [ENGLISH, CHINESE]
    """
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    words_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l
                   in lines[:num_examples]]
    return zip(*words_pairs)


def tokenize(lang):
    """Tokenize the language. Trans all vocab to index number.

    Args:
        lang: The lang texts.

    Returns:
        The tensor from texts and tokenizer.
    """
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # pad zero after sequences for the same length.
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    """Load dataset.

    Args:
        path: The texts file.
        num_examples: The max examples number.

    Returns:
        input tensor, target tensor, input tokenizer, target tokenizer.
    """
    inp_lang, target_lang = create_dataset(path, num_examples)
    inp_tensor, inp_tokenizer = tokenize(inp_lang)
    tar_tensor, tar_tokenizer = tokenize(target_lang)
    return inp_tensor, tar_tensor, inp_tokenizer, tar_tokenizer


def convert(lang_tokenizer, tensor):
    """Convert lang token to origin lang."""
    for t in tensor:
        if t != 0:
            print('%d -----> %s' % (t, lang_tokenizer.index_word[t]))


class Encoder(tf.keras.Model):
    """The encoder model."""

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        """The structure function.

        Args:
            vocab_size: The source vocabulary size.
            embedding_dim: The embedding dimension of vocabulary.
            enc_units: The units number of encoder layer.
            batch_size: The batch size.
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=0.5,
                                       recurrent_dropout=0.5)

    @tf.function
    def call(self, inputs, hidden_state, training=False):
        """The call function.

        Args:
            inputs: The inputs tensor.
            hidden_state: The hidden rnn layer initial state.
            training: If training.

        Returns:
            The output tensor and state tensor.
        """
        embed = self.embedding(inputs)
        outputs, state = self.gru(embed, initial_state=hidden_state)
        return outputs, state

    def initialize_hidden_state(self):
        """Initialize hidden rnn state."""
        return tf.zeros(shape=(self.batch_size, self.enc_units))


class BahandauAttention(tf.keras.layers.Layer):
    """The attention layer with bahandau score."""

    def __init__(self, units):
        """The structure function.

        Args:
            units: The units number of attention hidden layer.
        """
        super(BahandauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units=units)
        self.W2 = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    @tf.function
    def call(self, inputs, hidden_state):
        """The

        Args:
            inputs: The inputs.
            hidden_state: The rnn hidden layer state.

        Returns:
            The context vectors and attention weights.
        """
        # hidden state shape == (batch_size, hidden_size)
        # hidden_state_with_time_axis shape == (batch_size, 1, hidden_size)
        # inputs shape == (batch_size, seq_max_len, hidden_size)

        # Broadcast time axis for hidden state for calc the score.
        hidden_state_with_time_axis = tf.expand_dims(hidden_state, 1)

        # Score shape == (batch_size, seq_max_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(
            tf.nn.tanh(self.W1(hidden_state_with_time_axis) + self.W2(inputs)))

        # Attentions weights shape == (batch_size, seq_max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context vector shape == (batch_size, hidden_size)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector,
                                       1)  # Reduce sum in time axis.
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    """The decoder model."""

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        """The structure function.

        Args:
            vocab_size: The target language vocabulary size.
            embedding_dim: The embedding dimension.
            dec_units: The decoder hidden units.
            batch_size: The batch size.
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       dropout=0.5,
                                       recurrent_dropout=0.5)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention_layer = BahandauAttention(self.dec_units)

    @tf.function
    def call(self, inputs, hidden_state, enc_output, training=None):
        """The call function.

        Args:
            inputs: The inputs.
            hidden_state: The rnn hidden state.
            enc_output: The encoder output used to calc the attention weights.
            training: If training.

        Returns:
            outputs, hidden state, attention weights.
        """
        # enc_output shape == (batch_size, seq_max_len, hidden_size)

        # Calc the context vector.
        # The context vector shape == (batch_size, rnn_units)
        context_vector, attention_weights = self.attention_layer(enc_output,
                                                                 hidden_state)
        print('context_vector shape: {}'.format(context_vector.shape))

        # Decoder input shape == (batch_size, embed_dim)
        print('decoder input shape: {}'.format(inputs.shape))
        embed = self.embedding(inputs)
        # Embed of decoder input shape == (batch_size, 1, embed_dim)
        print('embed shape: {}'.format(embed.shape))

        # Concat context vector and embed of decoder input to attention vector.
        # The attention vector shape == (batch_size, 1, embed_dim + rnn_units)
        attention_vector = tf.concat([tf.expand_dims(context_vector, 1),
                                      embed], -1)
        print('attention_vector shape: {}'.format(attention_vector.shape))

        # Calc the output and state of gru.
        # The output shape == (batch_size, 1, hidden_size)
        output, state = self.gru(attention_vector)
        # Remove the time axis.
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab_size)
        output = self.fc(output)

        return output, state, attention_weights


def loss_function(real, pred):
    """Calc the loss.

    Args:
        real: The real value.
        pred: The prediction value.

    Returns:
        The loss.
    """
    crossentropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = crossentropy_loss(y_true=real, y_pred=pred)

    # Ignore 0.
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = loss * mask

    return tf.reduce_mean(loss)


@tf.function
def train_step(inp, target, hidden_state, encoder, decoder, optimizer, \
               tar_lang_tokenizer):
    """Training step.

    Args:
        inp: The input.
        target: The target.
        hidden_state: The hidden rnn state.
        encoder: The encoder model.
        decoder: The decoder model.
        optimizer: The optimizer to apply gradient.
        tar_lang_tokenizer: The target language tokenizer.

    Returns:
        The batch loss.
    """
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden_state = encoder(inp, hidden_state,
                                               training=True)
        dec_hidden_state = enc_hidden_state

        # The decoder input shape == (batch_size, 1)
        dec_inp = tf.expand_dims(
            [tar_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, target.shape[1]):
            dec_output, dec_hidden_state, _ = decoder(dec_inp,
                                                      dec_hidden_state,
                                                      enc_output,
                                                      training=True)
            loss += loss_function(real=target[:, t], pred=dec_output)
            # Use teacher forcing to training.
            dec_inp = tf.expand_dims(target[:, t], 1)
        batch_loss = loss / target.shape[1]
        training_variables = encoder.trainable_variables + \
                             decoder.trainable_variables
        gradients = tape.gradient(target=loss, sources=training_variables)
        optimizer.apply_gradients(zip(gradients, training_variables))

        global has_show_summary
        if not has_show_summary:
            encoder.summary()
            decoder.summary()
            has_show_summary = True
        return batch_loss


def train(encoder, decoder, dataset, steps_per_epoch, checkpoint,
          checkpoint_dir, tar_lang_tokenizer, epochs=EPOCHS):
    """Training model.

    Args:
        encoder: The encoder model.
        decoder: The decoder model
        dataset: The dataset iterator.
        steps_per_epoch: The steps number per epoch.steps_per_epoch
        tar_lang_tokenizer: The target language tokenizer.
        epochs: The total epochs number.

    """
    if os.path.exists(checkpoint_dir) \
            and tf.train.latest_checkpoint(checkpoint_dir):
        # Load checkpoint if exist.
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Tensorboard.
    summary_writer = tf.summary.create_file_writer(MODEL_PATH)

    with summary_writer.as_default():
        for epoch in range(epochs):
            begin_time = time.time()
            total_loss = 0
            hidden_state = encoder.initialize_hidden_state()
            tf.summary.trace_on(graph=True)

            for (batch, (inp, tar)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = train_step(inp, tar, hidden_state, encoder,
                                        decoder, optimizer, tar_lang_tokenizer)
                total_loss += batch_loss
                if batch % 100 == 0:
                    print(
                        'Epoch {} batch {}, loss: {:.4f}.'.format(epoch, batch,
                                                                  batch_loss))

            tf.summary.trace_export('nmt',
                                    step=epoch)
            tf.summary.scalar(name='batch_loss',
                              data=total_loss / steps_per_epoch,
                              step=epoch)

            checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {}, loss: {:.4f}.'.format(epoch, total_loss /
                                                   steps_per_epoch))
            print('Time cost {:.2f}s.'.format(time.time() - begin_time))
    summary_writer.close()
    return checkpoint


def translate(src_sentences, encoder, decoder, checkpoint, src_lang_tokenizer,
              tar_lang_tokenizer, max_length_inp, max_length_tar,
              checkpoint_dir):
    """Translate source sentence to target.

    Args:
        src_sentences: The source sentence list.
        encoder: The encoder.
        decoder: The decoder.
        checkpoint: The checkpoint.
        src_lang_tokenizer: The source language tokenizer.
        tar_lang_tokenizer: The target language tokenizer.
        max_length_inp: The max length of input sequences.
        max_length_tar: The max length of target sequences.
        checkpoint_dir: The checkpoint dir.

    Returns:
        The target sentences list.
    """
    # Load models.
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for i in range(len(src_sentences)):
        print('Source sentence: ' + src_sentences[i])
        inp_tensor = src_lang_tokenizer.texts_to_sequences([preprocess_sentence(
            src_sentences[i])])
        inp_tensor = tf.constant(inp_tensor, dtype=tf.float32)
        inp_tensor = tf.keras.preprocessing.sequence.pad_sequences(
            inp_tensor,
            maxlen=max_length_inp,
            padding='post')
        # print(inp_tensor)
        tar_sentence = ''
        hidden_state = tf.zeros(shape=(1, RNN_UNITS))
        enc_output, enc_hidden_state = encoder(inputs=inp_tensor,
                                               hidden_state=hidden_state,
                                               training=False)
        dec_hidden_state = enc_hidden_state
        dec_inp = tf.expand_dims([tar_lang_tokenizer.word_index[
                                       '<start>']], 1)
        for j in range(max_length_tar):
            dec_output, dec_hidden_state, attention_weights = decoder(
                inputs=dec_inp,
                hidden_state=dec_hidden_state,
                enc_output=enc_output,
                training=False)
            # print(dec_output)
            # print(attention_weights.shape)
            pred_word_index = np.argmax(dec_output.numpy()[0])
            pred_word = tar_lang_tokenizer.index_word[pred_word_index]
            if pred_word == '<end>':
                break
            tar_sentence = tar_sentence + ' ' + pred_word
            dec_inp = tf.expand_dims([pred_word_index], 1)
        print(tar_sentence)


if __name__ == '__main__':
    """
    en_sentence = u'The large crowd roared in approval as Mark Knopfler ' \
                  u'played ' \
                  'the first few bars of "Money for Nothing".'
    ch_sentence = u'就像马克·诺弗勒早期演唱的歌曲《金钱无用》一样，绝大多数的' \
                  u'人依然高呼赞成“金钱无用论”。'
    print(preprocess_sentence(en_sentence))
    print(preprocess_sentence(ch_sentence))

    en, ch = create_dataset(PATH_TO_FILE)
    print(en[-1])
    print(ch[-1])
    """

    inp_tensor, tar_tensor, inp_tokenizer, tar_tokenizer = load_dataset(
        PATH_TO_FILE)
    max_length_inp = inp_tensor.shape[1]
    max_length_tar = tar_tensor.shape[1]
    """
    print('source language tokenizer index_word: {}' \
          .format(inp_tokenizer.index_word))
    print('target language tokenizer index_word: {}' \
          .format(tar_tokenizer.index_word))
    print('source language tokens => words:')
    print(convert(inp_tokenizer, inp_tensor[-1]))
    print('target language tokens => word:')
    print(convert(tar_tokenizer, tar_tensor[-1]))

    # Split training and validation data.
    train_input_tensor, val_input_tensor, train_tar_tensor, val_tar_tensor = \
        train_test_split(inp_tensor, tar_tensor, test_size=0.2)
    print('training input tensor len: %d.' % len(train_input_tensor))
    print('training target tensor len: %d.' % len(train_tar_tensor))
    print('validation input tensor len: %d.' % len(val_input_tensor))
    print('validation target tensor len: %d.' % len(val_tar_tensor))
    """

    inp_vocab_size = len(inp_tokenizer.index_word) + 1  # Add index 0.
    tar_vocab_size = len(tar_tokenizer.index_word) + 1  # Add index 0.
    print('inp_vocab_size: ' + str(inp_vocab_size))
    print('tar_vocab_size: ' + str(tar_vocab_size))
    train_dataset = tf.data.Dataset \
        .from_tensor_slices((inp_tensor, tar_tensor)) \
        .shuffle(buffer_size=len(inp_tensor)) \
        .batch(BATCH_SIZE, drop_remainder=True)

    # Build model.
    encoder = Encoder(vocab_size=inp_vocab_size, embedding_dim=EMBEDDING_DIM,
                      enc_units=RNN_UNITS, batch_size=BATCH_SIZE)
    decoder = Decoder(vocab_size=tar_vocab_size, embedding_dim=EMBEDDING_DIM,
                      dec_units=RNN_UNITS, batch_size=BATCH_SIZE)

    # Create optimizer.
    optimizer = tf.keras.optimizers.Adam()

    # Create checkpoint.
    checkpoint_dir = os.path.join(MODEL_PATH, 'training_checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimzer=optimizer, encoder=encoder,
                                     decoder=decoder)

    steps_per_epoch = len(inp_tensor) // BATCH_SIZE
    train(encoder, decoder, train_dataset, steps_per_epoch, checkpoint,
          checkpoint_dir, tar_tokenizer, epochs=20)

    translate(['How\'s the weather today?',
               'You are fooled!',
               'If a person has not had a chance to acquire his target '
               'language by the time he\'s an adult, he\'s unlikely to be '
               'able to reach native speaker level in that language.'],
              encoder, decoder, checkpoint, inp_tokenizer, tar_tokenizer,
              max_length_inp, max_length_tar, checkpoint_dir)
