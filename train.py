
import tensorflow as tf
import numpy as np
import cv2
from utils import DataSet,read_test_set,read_train_sets,build_log_dir 
from model import ClothPredict
import time
from datetime import timedelta
import argparse
import os
from tqdm import tqdm,trange
import sys
import glob

def print_progress(sess, accuracy, iteration, feed_dict_train, feed_dict_validate, val_loss, log_path):
    # Calculate the accuracy on the training-set.
    acc = sess.run(accuracy, feed_dict=feed_dict_train)
    val_acc = sess.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Iteration {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(iteration, acc, val_acc, val_loss))

    with open(log_path + '/loss.csv', 'a') as f:
        f.write('%d, %.15f, %.15f, %.15f\n' % (iteration, acc, val_acc, val_loss))
        
"""
def sample_prediction(sess, y_pred_cls, test_im):

    
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
      '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',\
      '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', \
      '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', \
      '40', '41']])
        #y_true: np.array([[9,8,7,6,5,4,3,2,1,0]])
        #y_true: np.array([[0,0,0,0,0,0,0,0,0,0]])
        #y_true: np.array([[1,1,1,1,1,1,1,1,1,1]])
    }

    test_pred = sess.run(y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, help='Checkpoint to load all weights from.')
    parser.add_argument('--load-gen', type=str, help='Checkpoint to load generator weights only from.')
    parser.add_argument('--name', type=str, help='Name of experiment.')
    parser.add_argument('--overfit', action='store_true', help='Overfit to a single image.')
    parser.add_argument('--batch-size', type=int, default=16, help='Mini-batch size.')
    parser.add_argument('--log-freq', type=int, default=10000,
                        help='How many training iterations between validation/checkpoints.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for Adam.')
    parser.add_argument('--content-loss', type=str, default='mse', choices=['mse', 'L1','edge_loss_mse','edge_loss_L1'],
                        help='Metric to use for content loss.')
    parser.add_argument('--use-gan', action='store_true',
                        help='Add adversarial loss term to generator and trains discriminator.')
    parser.add_argument('--image-size', type=int, default=96, help='Size of random crops used for training samples.')
    parser.add_argument('--vgg-weights', type=str, default='vgg_19.ckpt',
                        help='File containing VGG19 weights (tf.slim)')
    parser.add_argument('--train-dir', type=str, help='Directory containing training images')
    parser.add_argument('--validate-benchmarks', action='store_true',
                        help='If set, validates that the benchmarking metrics are correct for the images provided by the authors of the SRGAN paper.')
    parser.add_argument('--gpu', type=str, default='0', help='Which GPU to use')
    parser.add_argument('--epoch', type=int, default='1000000', help='How many iterations ')
    parser.add_argument('--is-val', action='store_true', help='How many iterations ')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',\
      '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',\
      '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', \
      '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', \
      '40', '41']    
    # classes = ['1', '2', '3', '4', '5','6']
    num_classes = len(classes)
    '''Configuration and Hyperparameters'''
    # Convolutional Layer 1.
    filter_size1 = 3 
    num_filters1 = 32

    # Convolutional Layer 2.
    filter_size2 = 3
    num_filters2 = 32

    # Convolutional Layer 3.
    filter_size3 = 3
    num_filters3 = 64
    # Fully-connected layer.
    fc_size = 128  # Number of neurons in fully-connected layer.
    # Number of color channels for the images (RGB)
    num_channels = 3

    # image dimensions, square image -> img_size x img_size
    img_size = 28

    # Size of image when flattened to a single dimension
    img_size_flat = img_size * img_size * num_channels

    # Tuple with height and width of images used to reshape arrays.
    img_shape = (img_size, img_size)

    batch_size = 32
    validation_size = .16
    early_stopping = None  # use None if you don't want to implement early stoping


    # Create log folder
    if args.load and not args.name:
        log_path = os.path.dirname(args.load)
    else:
        log_path = build_log_dir(args, sys.argv)

    epochs = 1000
    train_path = 'done_dataset/train/'
    test_path = 'done_dataset/test/'
    checkpoint_dir = "models/"

    
    #test_images, test_ids = read_test_set(test_path, img_size)



    '''Placeholder variables'''
    x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
    x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1) 

    # Predicted class
    cloth_predict = ClothPredict(learning_rate=1e-4)
    layer_fc_out = cloth_predict.forward(x_image)
    y_pred = tf.nn.softmax(layer_fc_out)
    y_pred_cls = tf.argmax(y_pred, dimension=1) 

    #Loss
    cloth_predict_loss = cloth_predict.loss(layer_fc_out, y_true)
    optimizer = cloth_predict.optimizer(cloth_predict_loss)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        iteration = 0

        sess.run(tf.global_variables_initializer())

        train_batch_size = batch_size
        start_time = time.time()
        

        saver = tf.train.Saver()
        # Load all
        if args.load:
            iteration = int(args.load.split('-')[-1])
            saver.restore(sess, args.load)
            print(saver)
            print("load_process_DEBUG")

        if args.is_val:
            #pred = np.empty()
            files = sorted(glob.glob('./done_dataset/test/*.jpg'))
            file_name = list()
            pred_li = list()
            pbar = tqdm(range(len(files)),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i in pbar:
                img = cv2.resize(cv2.imread(files[i]),(img_size,img_size),cv2.INTER_LINEAR)/255.
                img = np.asarray(img)
                file_name.append(files[i].split('\\')[-1])            
                #print(files[i].split('\\')[-1])
                feed_dict_test = {
                    x: img.reshape(1, img_size_flat),
                    y_true: np.array([['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',\
                          '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',\
                          '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', \
                          '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', \
                          '40', '41']])

                }
                
                test_pred = sess.run(y_pred_cls, feed_dict=feed_dict_test)
                pred_li.append(test_pred)
                
            np.savetxt('prediction.txt', pred_li, fmt='%s', delimiter='\n')
            np.savetxt('file_name.txt',file_name, fmt='%s', delimiter='\n')
            print('__DEBUG__pred result length: %s' %len(pred_li))
            print('__DEBUG__pred result: %s' %pred_li)

                
        else:
            data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)
            print(data.train.num_examples)
                
            for i in range(epochs):
                pbar = tqdm(range(data.train.num_examples//batch_size),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

                for idx in pbar:
                    pbar.set_description('[Iteration: %s]'%iteration)

                    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
                    x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

                    x_batch = x_batch.reshape(train_batch_size, img_size_flat)
                    x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

                    feed_dict_train = {x: x_batch,
                                       y_true: y_true_batch}
                    
                    feed_dict_validate = {x: x_valid_batch,
                                          y_true: y_valid_batch}

                    sess.run(optimizer, feed_dict=feed_dict_train) 

                    if iteration % 1000 == 0: 
                        val_loss = sess.run(cloth_predict_loss, feed_dict=feed_dict_validate)
                        print_progress(sess, accuracy, iteration, feed_dict_train, feed_dict_validate, val_loss, log_path)

                        saver.save(sess, os.path.join(log_path, 'weights'), global_step=iteration, write_meta_graph=False)

                    iteration += 1 
                


if __name__ == "__main__":
    main()




