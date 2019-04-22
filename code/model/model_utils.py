import tensorflow as tf
import numpy as np

def masked_softmax_cross_entropy(predict, label, mask):
    '''
    Compute softmax cross-entropy loss with mask:
    Cross entropy: 1*ln(P(right_label))
    '''
    #Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=label)
    #Change the type of mask
    mask = tf.cast(mask, dtype=tf.float32)
    mask = mask/tf.reduce_mean(mask)
    #compute loss
    loss = loss*mask
    loss = tf.reduce_mean(loss)

    return loss

def masked_accuracy(predict, label, mask):
    '''
    Compute accuracy with mask
    '''

    #Calculate result and change the type of mask
    result = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    result = tf.cast(result, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    #Compute the right elements with mask
    result = result * mask

    accuracy = tf.reduce_sum(result)/tf.reduce_sum(mask)

    return accuracy

def early_stopping(acc_list, epochs, memory):
    '''
    Early-stopping
    '''
    if epochs < memory:
        return False

    if acc_list[-1] < np.mean(acc_list[-(memory + 1): -1]):
        return True
    else:
        return False




    
