#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2020/7/28 上午9:26
@Author : Catherinexxx
@Site : 
@File : bert_api.py.py
@Software: PyCharm
"""
from transformers import BertForMaskedLM,BertForSequenceClassification,BertTokenizer
import torch
from torch.utils.data import DataLoader

import sys, os
from bert_utils.classifier_args import classifier_args
from bert_utils.sequence_classifier import SequenceClassifier
from bert_utils.mlm_utils import FinetuneMlm, word_prediction_args
from bert_utils.utils import format_time, flat_accuracy, DataPrecessForSentence

from transformers import get_linear_schedule_with_warmup, AdamW

import logging
import pandas as pd
import csv
import time

class bert_tool:
    def __init__(self, model):
        self.model = model

        self.tokenizer = BertTokenizer.from_pretrained(self.model)
        self.masked_token = '[MASK]'
        self.sep_token = '[SEP]'
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                        else "cpu")
        self.model_name = 'BERT'
        self.seq = None

        handler = logging.StreamHandler()
        # handler.addFilter(logging.Filter('happytransformer'))

        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO,
            handlers=[handler]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info("Using model: %s", self.device)

    def predict_masked(self, text, masked_length=1):
        """
        :param text:  sentence contain [MASK]
        :return:
        """
        self.model = BertForMaskedLM.from_pretrained(self.model)

        text_tokens = self.tokenizer.tokenize(text)
        id_tokens = self.tokenizer.convert_tokens_to_ids(text_tokens)
        seg_ids = self._get_segment_ids(text_tokens)

        masked_index = text_tokens.index(self.masked_token)

        tokens_tensor = torch.tensor([id_tokens])
        segs_tensor = torch.tensor([seg_ids])

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(tokens_tensor, segs_tensor)[0]
            prediction = self._softmax(prediction)
            sub = ''
            for i in range(masked_index):
                predicted_index = torch.argmax(prediction[0, masked_index+i]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
                sub += predicted_token
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return sub

    def _softmax(self, value):
        # TODO: make it an external function
        return value.exp() / (value.exp().sum(-1)).unsqueeze(-1)

    def _get_segment_ids(self, tokenized_text: list):
        """
        Converts a list of tokens into segment_ids. The segment id is a array
        representation of the location for each character in the
        first and second sentence. This method only words with 1-2 sentences.
        Example:
        tokenized_text = ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]',
                          'jim', '[MASK]', 'was', 'a', 'puppet', '##eer',
                          '[SEP]']
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
        returns segments_ids
        """
        split_location = tokenized_text.index(self.sep_token)
        segment_ids = list()
        for i in range(0, len(tokenized_text)):
            if i <= split_location:
                segment_ids.append(0)
            else:
                segment_ids.append(1)
            # add exception case for XLNet

        return segment_ids

    def init_sequence_classifier(self):
        """
        Initializes a binary sequence classifier model with default settings
        """

        # TODO Test the sequence classifier with other models
        self.seq_args = classifier_args.copy()
        self.seq = BertForSequenceClassification.from_pretrained(
            self.model,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        if torch.cuda.is_available():
            self.seq.cuda()

        self.logger.info("A binary sequence classifier for %s has been initialized", self.model_name)


    def train_sequence_classifier(self, train_csv_path, eval_csv_path):
        """
        Trains the HappyTransformer's sequence classifier

        :param train_csv_path: A path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.
        """
        if not os.path.exists(self.seq_args['seq_model_dir']):
            os.makedirs(self.seq_args['seq_model_dir'])

        self.logger.info("***** Running Training *****")
        train_data = DataPrecessForSentence(self.tokenizer, train_csv_path)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=self.seq_args['batch_size'])
        validation_data = DataPrecessForSentence(self.tokenizer, eval_csv_path)
        validation_dataloader = DataLoader(validation_data, shuffle=True, batch_size=self.seq_args['batch_size'])

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = self.seq_args['num_epochs']
        optimizer = AdamW(self.seq.parameters(),
                          lr=self.seq_args['learning_rate'],  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=self.seq_args['adam_epsilon']  # args.adam_epsilon  - default is 1e-8.
                          )
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        # Store the average loss after each epoch so we can plot them.
        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            best_score = 0.0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            self.seq.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, _, b_labels = batch

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                self.seq.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = self.seq(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.seq.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))


            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.seq.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, _, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = self.seq(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask)

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

            # Report the final accuracy for this validation run.
            print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))
        if (eval_accuracy / nb_eval_steps) > best_score:
            best_score = (eval_accuracy / nb_eval_steps)
            torch.save({"epoch": epoch_i,
                        "model": self.seq.state_dict(),
                        "best_score": best_score},
                        os.path.join(self.seq_args['seq_model_dir'], "best.pth.tar"))

        print("")
        print("Training complete!")

    def eval_sequence_classifier(self, eval_csv_path):
        """
        Evaluates the trained sequence classifier against a testing set.

        :param csv_path: A path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.

        :return: A dictionary evaluation matrix
        """
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        self.logger.info("***** Running evaluation *****")



    def test_sequence_classifier(self, test_csv_path):
        """

        :param test_csv_path: a path to the csv evaluation file.
            Each test is contained within a row.
            The first column is for the the correct answers, either 0 or 1 as an int or a string .
            The second column is for the text.
        :return: A list of predictions where each prediction index is the same as the corresponding test's index
        """
        self.logger.info("***** Running Testing *****")



