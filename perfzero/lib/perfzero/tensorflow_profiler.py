# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Collect profiler data for Tensorboard with a separate thread."""

from __future__ import print_function

import logging
import os
import portpicker
import threading
import time
import traceback

import perfzero.utils as utils


class TensorFlowProfiler(object):
  """Collect profiler data for Tensorboard with a separate thread."""

  def __init__(self, profiler_enabled_time_str, output_dir):
    """Constructor.

    Args:
      profiler_enabled_time_str: the value of the config --profiler_enabled_time
      output_dir: log directory to place the profiler data
    """

    self.profiler_enabled_time_str = profiler_enabled_time_str
    self.output_dir = output_dir
    self.port = portpicker.pick_unused_port()
    self.thread_profiler = threading.Thread(target=self._on_profile)

    from tensorflow.python.profiler import profiler_v2 as profiler  # pylint: disable=g-import-not-at-top
    profiler.start_server(self.port)


  def _on_profile(self):
    if not self.profiler_enabled_time_str:
        return

    try:
        delay_ms = 1000 * int(self.profiler_enabled_time_str.split(':')[0].strip())
        duration_ms = 1000 * int(self.profiler_enabled_time_str.split(':')[1].strip())
    except ValueError:
        logging.error('Failed to parse --profiler_enabled_time: %s', self.profiler_enabled_time_str)
        return

    from tensorflow.python.profiler import profiler_client  # pylint: disable=g-import-not-at-top
    from tensorflow.python.profiler import profiler_v2 as profiler  # pylint: disable=g-import-not-at-top

    options = profiler.ProfilerOptions(
      host_tracer_level=2,
      python_tracer_level=0,
      device_tracer_level=0,
      delay_ms=delay_ms,
    )

    profiler_data_dir = os.path.join(self.output_dir, 'profiler_data')
    utils.make_dir_if_not_exist(profiler_data_dir)
    logging.info('Starting TensorFlow profiler and saving data to dir %s',
                 profiler_data_dir)

    profiler_client.trace('localhost:{}'.format(self.port), profiler_data_dir, duration_ms,
                        '', 100, options)

    logging.info('Started TensorFlow profiler')


  def start(self):
    """Schedule start/stop profiler event specified in profiler_enabled_time_str."""
    self.thread_profiler.start()

  def stop(self):
    """Stop scheduler and save profiler data if any event is cancelled."""
    self.thread_profiler.join()
