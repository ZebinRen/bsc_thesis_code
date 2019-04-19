import tensorflow as tf

def masked_softmax_cross_entropy(predict, label, mask):
    '''
    Compute softmax cross-entropy loss with mask:
    Cross entropy: 1*ln(P(right_label))
    '''
    #Compute loss
    loss = tf.nn.masked_softmax_cross_entropy(logits=predict, labels=label)
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
    mask = tf.cast(mask, dtype=tf.float32)

    #Compute the right elements with mask
    result = result * mask

    accuracy = tf.reduce_sum(result)/tf.reduce_sum(mask)

    



    
