import tensorflow as tf 
import numpy as np 


def my_input_fn():
    features = np.array([[5.9, 3.0, 4.2, 1.5],
                        [6.9, 3.1, 5.4, 2.1],
                        [5.1, 3.3, 1.7, 0.5]])


    labels = np.array([[1],[2],[0]])
    

    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.batch(1)
    iterator = ds.make_one_shot_iterator()
    feature, label = iterator.get_next()
    return feature, label


next_batch = my_input_fn()

with tf.Session() as sess:
    first_batch = sess.run(next_batch)
    second_batch = sess.run(next_batch)
    third_batch = sess.run(next_batch)
print (first_batch)
print (second_batch)
print (third_batch)