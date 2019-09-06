import chumpy as ch
import numpy as np

kpId2vertices = {
                 # 0: [38, 78, 79, 92, 108, 117, 118, 119, 120, 121, 122, 214, 215, 234, 239, 279],  #Wrist

                 # 1: [4, 229],  #ThumbP
                 # 2: [29, 105, 123, 126, 248, 286],  #ThumbMP
                 # 3: [707, 708, 709, 710, 711, 712, 713, 714, 753, 754],  #ThumbMT
                 4: [744],  #ThumbT

                 # 5: [137, 138, 168, 169, 172, 186, 258, 260, 263, 274],  #IndexP
                 # 6: [48, 49, 58, 59, 87, 156, 166, 167, 213, 225],  #IndexMP
                 # 7: [294, 295, 296, 297, 298, 299, 300, 301, 340, 341],  #IndexMT
                 8: [320],  #IndexT

                 # 9: [185, 187, 246, 262, 269, 270, 277, 370, 378, 379],  #MiddleP
                 # 10: [358, 359, 362, 363, 365, 373, 376, 377, 389, 394],  #MiddleMP
                 # 11: [406, 407, 408, 409, 410, 411, 412, 413, 452, 453],  #MiddleMT
                 12: [443],  #MiddleT

                 # 13: [76, 141, 160, 162, 197, 198, 247, 276, 290, 488],  # RingP
                 # 14: [470, 471, 474, 475, 477, 483, 486, 487, 499, 504],  # RingMP
                 # 15: [517, 518, 519, 520, 521, 522, 523, 524, 563, 564],  # RingMT
                 16: [555],  #RingT

                 # 17: [161, 199, 201, 202, 278, 289, 595, 604, 605],  #PinkP
                 # 18: [582, 583, 586, 587, 589, 599, 602, 603, 617, 622],  #PinkMP
                 # 19: [634, 635, 636, 637, 638, 639, 640, 641, 680, 681],  #PinkMT
                 20: [672]  #PinkT
                 }


def get_keypoints_from_mesh_ch(mesh_vertices, keypoints_regressed):
    """ Assembles the full 21 keypoint set from the 16 Mano Keypoints and 5 mesh vertices for the fingers. """
    keypoints = [0.0 for _ in range(21)] # init empty list

    # fill keypoints which are regressed
    mapping = {0: 0, #Wrist
               1: 5, 2: 6, 3: 7, #Index
               4: 9, 5: 10, 6: 11, #Middle
               7: 17, 8: 18, 9: 19, # Pinky
               10: 13, 11: 14, 12: 15, # Ring
               13: 1, 14: 2, 15: 3} # Thumb

    for manoId, myId in mapping.items():
        keypoints[myId] = keypoints_regressed[manoId, :]

    # get other keypoints from mesh
    for myId, meshId in kpId2vertices.items():
        keypoints[myId] = ch.mean(mesh_vertices[meshId, :], 0)

    keypoints = ch.vstack(keypoints)

    return keypoints

