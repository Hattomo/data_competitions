# -*- coding: utf-8 -*-

class TokenErrorRate():
    """
    Member of TER
    """

    def __init__(self, total_error: float, subtitle_error: float, delete_error: float, insert_error: float,
                 len_ref: float) -> None:
        self.total_error = total_error
        self.substitute_error = subtitle_error
        self.delete_error = delete_error
        self.insert_error = insert_error
        self.len_ref = len_ref
