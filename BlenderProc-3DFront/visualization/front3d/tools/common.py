# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

from collections import Counter

import numpy as np


class BaseDataset(object):
    """Implements the interface for all datasets that consist of rooms."""
    def __init__(self, rooms):
        assert len(rooms) > 0
        self.rooms = rooms

    def __len__(self):
        return len(self.rooms)

    @property
    def class_labels(self):
        raise NotImplementedError()

    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def object_types(self):
        raise NotImplementedError()

    @property
    def n_object_types(self):
        """The number of distinct objects contained in the rooms."""
        return len(self.object_types)

    @property
    def room_types(self):
        return set([rm.room_type for rm in self.rooms])

    @property
    def count_objects_in_rooms(self):
        return Counter([len(rm.bboxes) for rm in self.rooms])

    def post_process(self, s):
        return s

    @staticmethod
    def with_valid_room_ids(invalid_room_ids):
        def inner(room):
            return room if room.room_id not in invalid_room_ids else False
        return inner

    @staticmethod
    def with_room_ids(room_ids):
        def inner(room):
            return room if room.room_id in room_ids else False
        return inner

    @staticmethod
    def with_room(room_type):
        def inner(room):
            return room if room_type in room.room_type else False
        return inner

    @staticmethod
    def room_smaller_than_along_axis(max_size, axis=1):
        def inner(room):
            return room if room.bbox[1][axis] <= max_size else False
        return inner

    @staticmethod
    def room_larger_than_along_axis(min_size, axis=1):
        def inner(room):
            return room if room.bbox[0][axis] >= min_size else False
        return inner

    @staticmethod
    def floor_plan_with_limits(limit_x, limit_y, axis=[0, 2]):
        def inner(room):
            min_bbox, max_bbox = room.floor_plan_bbox
            t_x = max_bbox[axis[0]] - min_bbox[axis[0]]
            t_y = max_bbox[axis[1]] - min_bbox[axis[1]]
            if t_x <= limit_x and t_y <= limit_y:
                return room
            else:
                False
        return inner

    @staticmethod
    def with_valid_boxes(box_types):
        def inner(room):
            for i in range(len(room.bboxes)-1, -1, -1):
                if room.bboxes[i].label not in box_types:
                    room.bboxes.pop(i)
            return room
        return inner

    @staticmethod
    def without_box_types(box_types):
        def inner(room):
            for i in range(len(room.bboxes)-1, -1, -1):
                if room.bboxes[i].label in box_types:
                    room.bboxes.pop(i)
            return room
        return inner

    @staticmethod
    def with_generic_classes(box_types_map):
        def inner(room):
            for box in room.bboxes:
                # Update the box label based on the box_types_map
                box.label = box_types_map[box.label]
            return room
        return inner

    @staticmethod
    def with_valid_bbox_jids(invalid_bbox_jds):
        def inner(room):
            return (
                False if any(b.model_jid in invalid_bbox_jds for b in room.bboxes)
                else room
            )
        return inner

    @staticmethod
    def at_most_boxes(n):
        def inner(room):
            return room if len(room.bboxes) <= n else False
        return inner

    @staticmethod
    def at_least_boxes(n):
        def inner(room):
            return room if len(room.bboxes) >= n else False
        return inner

    @staticmethod
    def with_object_types(objects):
        def inner(room):
            return (
                room if all(b.label in objects for b in room.bboxes)
                else False
            )
        return inner

    @staticmethod
    def contains_object_types(objects):
        def inner(room):
            return (
                room if any(b.label in objects for b in room.bboxes)
                else False
            )
        return inner

    @staticmethod
    def without_object_types(objects):
        def inner(room):
            return (
                False if any(b.label in objects for b in room.bboxes)
                else room
            )
        return inner

    @staticmethod
    def filter_compose(*filters):
        def inner(room):
            s = room
            fs = iter(filters)
            try:
                while s:
                    s = next(fs)(s)
            except StopIteration:
                pass
            return s
        return inner


class BaseScene(object):
    """Contains all the information for a room."""
    def __init__(self, room_id, room_type, bboxes):
        self.bboxes = bboxes
        self.room_id = room_id
        self.room_type = room_type

    def __str__(self):
        return "Room: {} of type: {} contains {} bboxes".format(
            self.room_id, self.room_type, self.n_objects
        )

    @property
    def n_objects(self):
        """Number of bounding boxes / objects in a room."""
        return len(self.bboxes)

    @property
    def object_types(self):
        """The set of object types in this room."""
        return sorted(set(b.label for b in self.bboxes))

    @property
    def n_object_types(self):
        """Number of distinct objects in a room."""
        return len(self.object_types)

    def ordered_bboxes_with_centroid(self):
        centroids = np.array([b.centroid for b in self.bboxes])
        ordering = np.lexsort(centroids.T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes

    def ordered_bboxes_with_class_labels(self, all_labels):
        centroids = np.array([b.centroid for b in self.bboxes])
        int_labels = np.array([[b.int_label(all_labels)] for b in self.bboxes])
        ordering = np.lexsort(np.hstack([centroids, int_labels]).T)
        ordered_bboxes = [self.bboxes[i] for i in ordering]

        return ordered_bboxes
