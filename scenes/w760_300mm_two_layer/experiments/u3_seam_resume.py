"""Run the archived upper-3 tail from measured open-cup seam arm poses."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import two_row_physx
from source_hooks import replace_once


def with_seam_resume(source):
    start = source.index('            contact_plan_rotation = approach_front()')
    side_push = '                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)\n'
    end = source.index(side_push, start) + len(side_push)
    standard = source[start:end]
    resume = '''            if slot == 3 and "q_rad" in receipt:
                center, rotation = observe_resting_cube(
                    "upper_B_seam_resumed_pose", vision_camera, vision_estimator, stack_plane)
                for arm, saved in receipt["q_rad"].items():
                    q = np.asarray(saved)
                    robots[arm].set_joint_positions(q, joint_indices=joints[arm])
                    robots[arm].set_joint_velocities(np.zeros(7), joint_indices=joints[arm])
                    command_positions(arm, q)
                set_contact_drive_mode(True)
                for arm, target in receipt["q_target_rad"].items():
                    command_positions(arm, np.asarray(target))
                controller = robots[push_side].get_articulation_controller()
                stiffness, damping = (np.asarray(values).copy() for values in controller.get_gains())
                indices = joints[push_side]
                stiffness[indices] *= .35
                damping[indices] *= math.sqrt(.35)
                controller.set_gains(kps=stiffness, kds=damping)
                allowed_contacts.update(robots)
                contact_monitor.phase = "upper_B_seam_pose_resume"
                record(contact_monitor.phase, "ARCHIVED_OPEN_SEAM_POSE_RESTORED", source_step=receipt["source_step"])
                step(12)
                check_current_collision(contact_monitor.phase)
                side_offset = np.array([-push_sign * half, 0., .010])
                retouch_after_support_seam(center, rotation, side_offset)
                push_to_boundary(push_side, 0, push_sign, "upper_B_side", side_target, follow=support_side)
            else:
'''
    return replace_once(source, standard, resume + "".join("    " + line for line in standard.splitlines(keepends=True)))


if __name__ == "__main__":
    build = two_row_physx.build
    two_row_physx.build = lambda: with_seam_resume(build())
    two_row_physx.main()
