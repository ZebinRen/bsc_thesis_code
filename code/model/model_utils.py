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
    
