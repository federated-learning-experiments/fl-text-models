import tensorflow as tf
import tensorflow_federated as tff

def learn_from_pretrained_model(iterative_process, pretrained_model):
    """
    All of the values of the pre-trained model's trainable weights are transferred to the TFF model
    to be fine-tuned.
    """
    model_to_finetune = iterative_process.initialize()
    for l1 in range(len(model_to_finetune.model[0])):
        for l2 in range(len(model_to_finetune.model[0][l1])):
            model_to_finetune.model[0][l1][l2] = pretrained_model.trainable_weights[l1].numpy()[l2]

    return model_to_finetune