# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import typing as tp

from .events import Event


class EventTypesHelper:
    """Computes and stores information about the event types
    provided either as an actual type, or a type name, or a tuple of type names
    to get a unified and simple access while the event type can be specified
    in multiple ways.

    Parameter
    ---------
    event_types: Event type, or str, or tuple of str
        event type or name of an event or tuple of names of events

    Attributes
    ----------
    classes: tuple of Event types
        the classes specified as event types (as a tuple even if only 1 type was specified)
    names: str
        the list of event type names specified, including subclasses. This is particularly
        handy to filter a dataframe: :code:`events[events.type.isin(helper.names)]`
    """

    def __init__(self, event_types: str | tp.Type[Event] | tp.Sequence[str]) -> None:
        self.specified = event_types
        if inspect.isclass(event_types):
            self.classes: tp.Tuple[tp.Type[Event], ...] = (event_types,)
        else:
            if isinstance(event_types, str):
                event_types = (event_types,)
            try:
                self.classes = tuple(Event._CLASSES[x] for x in event_types)  # type: ignore
            except KeyError as e:
                avail = list(Event._CLASSES)
                msg = f"{event_types} is an invalid event name, use one of {avail}"
                raise ValueError(msg) from e
        items = Event._CLASSES.items()
        self.names = [x for x, y in items if issubclass(y, self.classes)]
