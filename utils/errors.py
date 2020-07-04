#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:09:02 2020

@author: alain
"""

import traceback

# *****************************************************************************************************************************
# Wrapping animation exception

def error_header(*args):

    msg  = "\n\n" + "="*80 + "\n"
    msg += "Animation wrapper error\n\n"

    title = args[0] if args else "Undocument error"
    msg += "::: {}\n\n".format(title)

    if args:
        for i in range(1, len(args)):
            msg += "{}\n".format(args[i])

    return msg

class WrapException(Exception):
    def __init__(self, title, *args):
        self.message = error_header(title, *args)

        print(traceback.format_exc())

    def __str__(self):
        return self.message


class WrapShapeException(Exception):
    def __init__(self, given_shape, expected_shape, *args):
        title = "Arrays shapes mismatch"
        msg   = "Shape given is '{}' but '{}' is expected".format(given_shape, expected_shape)
        self.message = error_header(title, msg, *args)

        #traceback.print_stack()
        print(traceback.format_exc())

    def __str__(self):
        return self.message


# *****************************************************************************************************************************
# Errors message

def error(title, obj=None, message=None):

    raise WrapException(title, obj, message)

    print("-"*100)
    print("Title")
    if obj is not None:
        if hasattr(obj, 'dump'):
            obj.dump()
        else:
            print(obj)
    print("-"*100)

    if message is not None:
        raise NameError(message)
