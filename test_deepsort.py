import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from movement import MovementAnalyzer


# --------------------------------------------------
# 1. Initialize DeepSORT
# --------------------------------------------------

tracker = DeepSort(
    max_age=30,
    n_init=1
)


# --------------------------------------------------
# 2. Initialize movement analyzer
# --------------------------------------------------

movement_analyzer = MovementAnalyzer()


# --------------------------------------------------
# 3. Create blank test frame
# --------------------------------------------------

frame = np.zeros(
    (720, 1280, 3),
    dtype=np.uint8
)


# --------------------------------------------------
# 4. Three simulated animals
#
# DeepSORT input:
# [left, top, width, height]
# --------------------------------------------------

frames = [

    # --------------------------------------------------
    # Frame 1
    # --------------------------------------------------
    [
        # Elephant: small
        ([200, 250, 100, 125], 0.91, "elephant"),

        # Deer: constant size
        ([500, 250, 120, 150], 0.87, "deer"),

        # Monkey: large
        ([800, 200, 200, 250], 0.84, "monkey")
    ],

    # --------------------------------------------------
    # Frame 2
    # --------------------------------------------------
    [
        # Elephant: getting larger
        ([190, 240, 120, 150], 0.91, "elephant"),

        # Deer: same size
        ([500, 250, 120, 150], 0.87, "deer"),

        # Monkey: getting smaller
        ([810, 210, 180, 225], 0.84, "monkey")
    ],

    # --------------------------------------------------
    # Frame 3
    # --------------------------------------------------
    [
        # Elephant: getting larger
        ([180, 230, 145, 180], 0.91, "elephant"),

        # Deer: same size
        ([500, 250, 120, 150], 0.87, "deer"),

        # Monkey: getting smaller
        ([820, 220, 160, 200], 0.84, "monkey")
    ],

    # --------------------------------------------------
    # Frame 4
    # --------------------------------------------------
    [
        # Elephant: getting larger
        ([165, 215, 175, 220], 0.91, "elephant"),

        # Deer: same size
        ([500, 250, 120, 150], 0.87, "deer"),

        # Monkey: getting smaller
        ([830, 230, 140, 175], 0.84, "monkey")
    ],

    # --------------------------------------------------
    # Frame 5
    # --------------------------------------------------
    [
        # Elephant: getting larger
        ([150, 200, 210, 265], 0.91, "elephant"),

        # Deer: same size
        ([500, 250, 120, 150], 0.87, "deer"),

        # Monkey: getting smaller
        ([840, 240, 120, 150], 0.84, "monkey")
    ]
]


# --------------------------------------------------
# 5. Process every frame
# --------------------------------------------------

for frame_number, detections in enumerate(frames, start=1):

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    print(f"\nFrame {frame_number}")

    for track in tracks:

        # Get bounding box
        bbox = track.to_ltrb()

        # Calculate movement
        movement = movement_analyzer.update(
            track.track_id,
            bbox
        )

        print(
            "Track ID:",
            track.track_id,
            "| Confirmed:",
            track.is_confirmed(),
            "| Class:",
            track.get_det_class(),
            "| Movement:",
            movement,
            "| BBox:",
            bbox
        )