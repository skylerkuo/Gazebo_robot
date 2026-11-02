"""ACT followed by official CheatCode geometry; all accepted commands are recorded."""
from aic_example_policies.ros.CheatCode import CheatCode
from rclpy.time import Time
import numpy as np

from automatic_expert_control import (AutomaticExpertOperator,InPlaceExpertAlignment,
    EXPERT_CONTROL_VERSION,EXPERT_MAX_SECONDS)
from ftact_control import pose_array
from policies.HumanFTACT import HumanFTACT


class AutoExpertFTACT(HumanFTACT):
    recording_controller='model_and_official_expert'
    recording_control_version=EXPERT_CONTROL_VERSION

    def create_operator(self):
        self.recording_control_parameters={'expert_z_stiffness':self.settings.get('expert_z_stiffness',600.),
            'expert_z_damping':self.settings.get('expert_z_damping',70.),
            'model_stiffness':list(self.settings['stiffness']),'model_damping':list(self.settings['damping'])}
        return AutomaticExpertOperator(self.settings['expert_after_seconds'],self.settings.get('expert_max_seconds',EXPERT_MAX_SECONDS))

    def correction_source(self):return 'expert'

    def motion(self,command):
        motion=super().motion(command)
        if self.operator.mode=='human':
            # Keep ACT's original controller during its attempt. Expert recovery
            # needs enough axial stiffness to follow a bounded insertion target.
            stiffness=list(self.settings['stiffness']);damping=list(self.settings['damping'])
            stiffness[2]=self.settings.get('expert_z_stiffness',600.)
            damping[2]=self.settings.get('expert_z_damping',70.)
            motion.target_stiffness=np.diag(stiffness).astype(float).flatten().tolist()
            motion.target_damping=np.diag(damping).astype(float).flatten().tolist()
        return motion

    def insert_cable(self,task,get_observation,move_robot,send_feedback):
        # The official engine reuses this policy/model across trials.
        self.operator=self.create_operator()
        self.teacher=CheatCode(self._parent_node)
        self.teacher._task=task
        self.expert_start=None;self.port=None;self.alignment=None;self.expert_phase=None
        self.correction_details={}
        return super().insert_cable(task,get_observation,move_robot,send_feedback)

    def correction_target(self,previous,control,ticks):
        now=self.time_now().nanoseconds*1e-9
        def transform_pose(transform):
            return [transform.translation.x,transform.translation.y,transform.translation.z,
                transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]
        buffer=self._parent_node._tf_buffer
        if self.expert_start is None:
            self.expert_start=now
            task=self.teacher._task
            self.port=buffer.lookup_transform('base_link',
                f'task_board/{task.target_module_name}/{task.port_name}_link',Time()).transform
            self.write({'type':'expert_start','ground_truth_supervision':True,
                'model_ground_truth_inputs':False,'simulation_time':now,
                'teacher':'official CheatCode.calc_gripper_pose',
                'controller_version':EXPERT_CONTROL_VERSION,
                'expert_z_stiffness':self.settings.get('expert_z_stiffness',600.),
                'expert_z_damping':self.settings.get('expert_z_damping',70.),
                'schedule':'align XY/rotation at takeover height; advance Z reference 3mm/s to tip depth limit; force pause; never lift'})
        actual=transform_pose(buffer.lookup_transform('base_link','gripper/tcp',Time()).transform)
        task=self.teacher._task
        plug=transform_pose(buffer.lookup_transform('base_link',f'{task.cable_name}/{task.plug_name}_link',Time()).transform)
        if self.alignment is None:self.alignment=InPlaceExpertAlignment(actual,now)
        official=pose_array(self.teacher.calc_gripper_pose(self.port,
            slerp_fraction=1.,position_fraction=1.,z_offset=0.,reset_xy_integrator=self.expert_phase is None))
        target,details=self.alignment.target(official,actual,plug,transform_pose(self.port),now,force_n=control.get('force_n'))
        self.correction_details=dict(details,expert_controller_version=EXPERT_CONTROL_VERSION,
            expert_z_stiffness=self.settings.get('expert_z_stiffness',600.),
            expert_z_damping=self.settings.get('expert_z_damping',70.))
        if details['phase']!=self.expert_phase:
            self.expert_phase=details['phase']
            self.write({'type':'expert_phase','simulation_time':now,**details})
        return target
