"""Official CheatCode teacher with the shared synchronized four-camera recorder."""
import json
import os
from pathlib import Path

import cv2
from aic_example_policies.ros.CheatCode import CheatCode
from sideview_data import TASK
from sideview_recorder import Recorder


class CollectSFP(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        cv2.setNumThreads(1)
        self.settings = json.loads(Path(os.environ['SIDEVIEW_SETTINGS']).read_text())
        if not hasattr(parent_node, '_sideview_recorder'):
            parent_node._sideview_recorder = Recorder(parent_node, self.settings)
        self.recorder = parent_node._sideview_recorder

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        if (task.target_module_name, task.plug_name, task.port_name) != (
                TASK['target_module'], TASK['plug'], TASK['port']):
            raise ValueError('Only SFP0 collection is supported')
        self.recorder.begin(task)
        returned = False
        def recorded_move(motion_update=None, joint_motion_update=None):
            try:
                result = move_robot(motion_update=motion_update, joint_motion_update=joint_motion_update)
                if result is False or motion_update is None:
                    raise RuntimeError('Teacher command was rejected')
                if self.settings.get('recording_mode') == 'official_cheatcode_only':
                    self.recorder.command(motion_update, source='expert', segment=0, human_active=True)
                else:
                    self.recorder.command(motion_update)
                return result
            except Exception as error:
                # The official set_pose_target catches command exceptions. Preserve the
                # failure so an accepted but unrecorded command can never pass import.
                self.recorder.recording_error = f'{type(error).__name__}: {error}'
                raise
        try:
            returned = super().insert_cable(task, get_observation, recorded_move, send_feedback)
            return returned
        finally:
            self.recorder.finish(returned, controller=(
                'official_cheatcode_only' if self.settings.get('recording_mode') == 'official_cheatcode_only'
                else 'teacher'))

