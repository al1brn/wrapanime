#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 19:24:03 2020

@author: alain
"""
import inspect

import numpy as np

from wrapanime.utils.errors import WrapException

def is_array(obj):
    return hasattr(obj, '__len__')

# ---------------------------------------------------------------------------

def for_each(alist, f, **kwargs):
    """Call a function for each item in the loop

    The template of the function is f(item, **kwargs)

    The item argument come from the list.
    If one of the arguments is 'index', it takes the index value of the loop.

    The other arguments of f are taken from the kwargs of for_each with one
    particularity:

    If a f argument 'arg' is not in the for_each kwargs dictionnatory, the
    algorithm can still feed it to f under the following conditions:
    - 'args' is present in the for_each kwargs
    - 'args' is a list type argument
    - the length of 'args' is equal to the list length

    Exemple
    -------

    def f(item, index, factor, location):
        # Loop index is index
        item.factor   = factor
        item.location = location

    # objects is a collection of objects
    # locs is a collection of vectors

    for_each(objects, f, factor=1, locations=locs)

    Is equivalent to:

    for index, item in enumerate(alist):
        f(item, index=index, factor=factor, location=locs[index])

    Parameters
    ----------
    alist: array like of itmes
        List to loop on
    f: function of template f(item [, index], **kwargs])
        The function to call
    **kwargs: keywaord arguments
        Arguments to use in each function call

    Returns
        None
    """
    
    if alist is None:
        return

    # Get the full argument specifications
    required_args = inspect.getfullargspec(f)

    # Prepare the dictionnary to call the function f
    # First item is iterated alist
    # Index 0 means the item comes from the first list in zips
    call_args = {required_args.args[0]: 0}

    # Arrays to include in the iterations loop
    zips = [alist]
    zip_args = [required_args.args[0]]

    # A required argument can have three sources:
    # - index is fed by the for_each loop itself
    # - name is fed by the argument of the same name in kwargs
    # - name is fed by the iterated item in the array names in kwargs

    def an_argument(num, name):

        # Done at initialization time
        if num == 0:
            return

        # Index will be fed by the loop index
        if name == 'index':
            call_args[name] = None
            return True

        # Simple if the name is in kwargs
        if name in kwargs:
            call_args[name] = kwargs[name]
            return

        # The argument must have a default value
        # If not it must be the singular of an list name passed in kwargs
        def_index_start = len(required_args.args) - len(required_args.defaults)
        def_index = num - def_index_start

        # The name must be the singular of a list name
        list_name = name+'s'
        if not list_name in kwargs:
            if def_index < 0:
                raise WrapException("Argument '{}' with no default value: impossible to call the function '{}'".format(name, f.__name__))
            call_args[name] = required_args.defaults[def_index]
            return True

        # Is the list really a list
        if not is_array(kwargs[list_name]):
            raise WrapException("The argument '{}' must be a list of length {} to match the required argument '{}' in function '{}'".format(
                    list_name, len(alist), name, f.__name__))


        # The length of the list must be of the same length
        # If the length is different, it is certainly due to a user mistake
        if len(kwargs[list_name]) != len(alist):
            error_header("Length {} of list '{}' must be the same as the main list: {}".format(
                    len(kwargs[list_name]), list_name, len(alist)))

        # At last :-)

        call_args[name] = None
        zips.append(kwargs[list_name])
        zip_args.append(name)
        return True

    # Loop on the arguments
    feed_index = False
    for num, name in enumerate(required_args.args):
        if name == 'index':
            feed_index = True
        an_argument(num, name)

    # Loop
    for index, vals in enumerate(zip(*zips)):
        if feed_index:
            call_args['index'] = index
        for i in range(len(vals)):
            call_args[zip_args[i]] = vals[i]

        f(**call_args)


# ---------------------------------------------------------------------------
# Read attr from a list of items

def get_attr(alist, name, value_type='f'):
    """Read attributes from a list of items.

    Parameters
    ----------
    alist: array like
        List of objects to read the attribute from
    name: str
        The name of the attribute to read
    value_type: numpy type
        Type for the returned array

    Returns
    -------
    numpy array
        Array of the same length as alist with the read attributes
    """

    values = np.empty(len(alist), value_type)
    for i in range(len(alist)):
        values[i] = getattr(alist[i], name)
    return values

# ---------------------------------------------------------------------------
# Write attr to a list of items

def set_attr(alist, name, values):
    """Write attributes to a list of items.

    Parameters
    ----------
    alist: array like
        List of objects to write the attribute to
    name: str
        The name of the attribute to write
    values: array like
        The values of the attribute to write
    """

    for item, value in zip(alist, values):
        setattr(item, name, value)
