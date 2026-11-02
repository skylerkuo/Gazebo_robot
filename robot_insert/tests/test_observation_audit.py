"""Distance units and observation-only neighbor eligibility for the ACT audit."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_sfp0_observations import errors,neighbor_pairs,balanced_train_anchors


class ObservationAuditTests(unittest.TestCase):
    def example(self):
        target=np.zeros((3,20,7));target[...,6]=1;target[1,:,2]=.02;target[2,:,2]=.04
        emb=np.zeros((3,3,4));emb[...,0]=1
        return dict(split=np.array(['train','train','validation']),episode=np.array([1,2,3]),
            source=np.array([11,12,13]),frame=np.zeros(3,dtype=int),state=np.zeros((3,6)),
            wrench=np.zeros((3,6)),embedding=emb,rgb_small=np.zeros((3,3,48,48,3),dtype=np.uint8),
            target=target,prediction=target.copy())

    def limits(self):
        return dict(joint_max_rad=.005,force_l2_n=1.,torque_l2_nm=.1,camera_cosine_min=.995,camera_mae_max_255=3.)

    def test_position_units_and_quaternion_sign_invariance(self):
        values=self.example();values['prediction'][...,:3]+=[.003,.004,0.]
        values['prediction'][...,3:]*=-1
        position,rotation,delta=errors(values)
        np.testing.assert_allclose(position,5.)
        np.testing.assert_allclose(rotation,0.)
        np.testing.assert_allclose(delta[0,0],[3.,4.,0.],atol=1e-10)

    def test_neighbors_use_observations_then_measure_targets(self):
        v=self.example()
        rows,summary=neighbor_pairs(v,'train_cross_episode',self.limits())
        self.assertEqual(summary['pairs'],1)
        self.assertAlmostEqual(rows[0]['mean_chunk_target_diff_mm'],20.)
        v['target'][:]=0
        rows2,_=neighbor_pairs(v,'train_cross_episode',self.limits())
        self.assertEqual([(r['a'],r['b']) for r in rows],[(r['a'],r['b']) for r in rows2])
        v['rgb_small'][1,2]=100
        self.assertEqual(neighbor_pairs(v,'train_cross_episode',self.limits())[1]['pairs'],0)

    def test_within_clip_pairs_require_one_second_separation(self):
        v=self.example();v['split'][:]='validation';v['episode'][:]=1;v['frame'][:]=[0,10,20]
        rows,_=neighbor_pairs(v,'validation_within_episode',self.limits())
        self.assertEqual([(r['frame_a'],r['frame_b']) for r in rows],[(0,20)])

    def test_train_anchor_sampling_is_unique_balanced_and_within_segment(self):
        index=dict(segments=[dict(split='train',episode_index=i,start=3,end=100) for i in range(7)])
        anchors=balanced_train_anchors(index,count=30)
        self.assertEqual(anchors,balanced_train_anchors(index,count=30))
        self.assertEqual(len(set(anchors)),30)
        counts=[sum(e==i for e,f in anchors) for i in range(7)]
        self.assertLessEqual(max(counts)-min(counts),1)
        self.assertTrue(all(3<=f<80 for e,f in anchors))


if __name__=='__main__':unittest.main()
