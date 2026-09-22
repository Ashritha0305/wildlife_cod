class MovementAnalyzer:

    def __init__(
        self,
        stationary_threshold=3.0,
        approaching_threshold=0.05
    ):
        # Stores previous information for every tracking ID
        self.history = {}

        # Threshold for deciding whether an object barely moved
        self.stationary_threshold = stationary_threshold

        # Threshold for bounding-box size change
        self.approaching_threshold = approaching_threshold


    def update(self, track_id, bbox):

        # ---------------------------------------------
        # 1. Extract bounding-box coordinates
        # ---------------------------------------------

        x1, y1, x2, y2 = bbox


        # ---------------------------------------------
        # 2. Calculate center of the animal
        # ---------------------------------------------

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2


        # ---------------------------------------------
        # 3. Calculate bounding-box size
        # ---------------------------------------------

        width = x2 - x1
        height = y2 - y1

        area = width * height


        # ---------------------------------------------
        # 4. First time seeing this tracking ID
        # ---------------------------------------------

        if track_id not in self.history:

            self.history[track_id] = {
                "center_x": center_x,
                "center_y": center_y,
                "area": area
            }

            return "stationary"


        # ---------------------------------------------
        # 5. Get previous information
        # ---------------------------------------------

        previous = self.history[track_id]


        # ---------------------------------------------
        # 6. Calculate movement of the center
        # ---------------------------------------------

        dx = center_x - previous["center_x"]
        dy = center_y - previous["center_y"]


        # ---------------------------------------------
        # 7. Calculate bounding-box area change
        # ---------------------------------------------

        previous_area = previous["area"]

        if previous_area > 0:

            area_change = (
                area - previous_area
            ) / previous_area

        else:

            area_change = 0


        # ---------------------------------------------
        # 8. Update history
        # ---------------------------------------------

        self.history[track_id] = {
            "center_x": center_x,
            "center_y": center_y,
            "area": area
        }


        # ---------------------------------------------
        # 9. Calculate how far the center moved
        # ---------------------------------------------

        movement_distance = (
            dx ** 2 + dy ** 2
        ) ** 0.5


        # ---------------------------------------------
        # 10. Approach / retreat based on bbox scale
        # ---------------------------------------------

        if area_change > self.approaching_threshold:

            return "approaching"


        if area_change < -self.approaching_threshold:

            return "moving away"


        # ---------------------------------------------
        # 11. Stationary check
        # ---------------------------------------------

        if movement_distance < self.stationary_threshold:

            return "stationary"


        # ---------------------------------------------
        # 12. Horizontal movement
        # ---------------------------------------------

        if abs(dx) > abs(dy):

            if dx > 0:

                return "moving right"

            else:

                return "moving left"


        # ---------------------------------------------
        # 13. Vertical movement
        # ---------------------------------------------

        else:

            if dy > 0:

                return "moving down"

            else:

                return "moving up"