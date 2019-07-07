import tensorflow as tf

from six.moves import cPickle

from model import Model
import argparse
import falcon
import os


class Generator:
    def load(self, args):
        self.args = args
        with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
            words, vocab = cPickle.load(f)
            self.words = words
            self.vocab = vocab
        self.model = Model(saved_args, True)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def generate(self, count=12, length=150):
        args = self.args

        return [self.model.sample(
            self.sess, self.words, self.vocab, length, args.prime, args.sample, args.pick, args.width, args.quiet
        ) for _ in range(count)]


class StatusResource:
    def on_get(self, req, res):
        res.media = {
            "status": "ok"
        }


class GenerateResource:
    def on_get(self, req, res):
        count = 12
        length = 150
        if req.params.get('count'):
            count = min(int(req.params.get('count')), 50)
        if req.params.get('length'):
            length = min(int(req.params.get('length')), 1000)

        res.media = generator.generate(count=count, length=length)


generator = Generator()
api = falcon.API()
api.add_route('/status', StatusResource())
api.add_route('/generate', GenerateResource())

if __name__ == '__main__':
    from wsgiref import simple_server

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to load stored checkpointed models from')
    parser.add_argument('--prime', type=str, default=' ',
                        help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                        help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                        help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--quiet', '-q', default=False, action='store_true',
                        help='suppress printing the prime text (default false)')

    parser.add_argument('--port', type=int, default=9000)
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    generator.load(args)
    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
