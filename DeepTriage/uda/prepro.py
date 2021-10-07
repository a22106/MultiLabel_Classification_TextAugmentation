from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
from absl import app
from absl import flags

import numpy as np
#import tensorflow as tf
import tensorflow as tf

# from augmentation import aug_policy
from augmentation import sent_level_augment
from augmentation import word_level_augment
from utils import raw_data_utils
from utils import tokenization

FLAGS = flags.FLAGS

def main(_):


  if FLAGS.max_seq_length > 512:
    raise ValueError(
        "Cannot use sequence length {:d} because the BERT model "
        "was only trained up to sequence length {:d}".format(
            FLAGS.max_seq_length, 512))

  processor = raw_data_utils.get_processor(FLAGS.task_name)
  # Create tokenizer
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  if FLAGS.data_type == "sup":
    sup_out_dir = FLAGS.output_base_dir
    tf.logging.info("Create sup. data: subset {} => {}".format(
        FLAGS.sub_set, sup_out_dir))

    proc_and_save_sup_data(
        processor, FLAGS.sub_set, FLAGS.raw_data_dir, sup_out_dir,
        tokenizer, FLAGS.max_seq_length, FLAGS.trunc_keep_right,
        FLAGS.worker_id, FLAGS.replicas, FLAGS.sup_size,
    )
  elif FLAGS.data_type == "unsup":
    assert FLAGS.aug_ops is not None, \
        "aug_ops is required to preprocess unsupervised data."
    unsup_out_dir = os.path.join(
        FLAGS.output_base_dir,
        FLAGS.aug_ops,
        str(FLAGS.aug_copy_num))
    data_stats_dir = os.path.join(FLAGS.raw_data_dir, "data_stats")


    tf.logging.info("Create unsup. data: subset {} => {}".format(
        FLAGS.sub_set, unsup_out_dir))
    proc_and_save_unsup_data(
        processor, FLAGS.sub_set,
        FLAGS.raw_data_dir, data_stats_dir, unsup_out_dir,
        tokenizer, FLAGS.max_seq_length, FLAGS.trunc_keep_right,
        FLAGS.aug_ops, FLAGS.aug_copy_num,
        FLAGS.worker_id, FLAGS.replicas)


if __name__ == "__main__":
  app.run(main)