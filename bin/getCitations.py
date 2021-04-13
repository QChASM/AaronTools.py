#!/usr/bin/env python3
import ast
import dis
import importlib
import inspect
import os
import re
from importlib import resources
from pprint import pprint

from AaronTools import getlogger

LOG = getlogger()


def main(args):
    modules = [
        r[:-3] for r in resources.contents(args.package) if r.endswith(".py")
    ]

    obj = None
    class_name, function_name = None, None
    if os.path.isfile(args.name):
        with open(args.name) as f:
            obj = compile(f.read(), filename=args.name, mode="exec")
    else:
        match = re.match("(\w+)\.(\w+)", args.name)
        if match is not None:
            class_name, function_name = match.groups()
        else:
            match = re.match("(\w+)", args.name)
            if match is not None:
                class_name, function_name = match.group(1), None
        for module in modules:
            module_name = module
            module = "{}.{}".format(args.package, module_name)
            try:
                module = importlib.import_module(module)
            except Exception:
                continue
            all_classes = inspect.getmembers(module, inspect.isclass)
            for c in all_classes:
                if class_name == c[0]:
                    obj = c[1]
                    break
            else:
                continue
            break
        if function_name is not None:
            obj = getattr(obj, function_name)
    if obj is None and class_name is not None:
        if function_name is None:
            LOG.error("Cannot find %s in %s", class_name, args.package)
        else:
            LOG.error(
                "Cannot find %s.%s in %s",
                class_name,
                function_name,
                args.package,
            )
        exit(1)
    elif obj is None:
        LOG.error("Cannot load %s", args.name)
        exit(1)
    for citation in set(get_citations(obj, args.name)):
        print(*citation)


def get_citations(obj, obj_name, done=None, citations=None):
    if done is None:
        done = set([])
    if citations is None:
        citations = []
    if obj in done:
        return citations
    done.add(obj)

    try:
        instructions = [inst for inst in dis.get_instructions(obj)]
    except TypeError:
        return citations
    names = {}
    methods = {}
    add_obj = set([])
    for i, inst in enumerate(instructions):
        if (
            inst.argval == "CITATION"
            and instructions[i - 1].opname == "LOAD_CONST"
        ):
            names.setdefault(inst.argval, [])
            names[inst.argval].append((obj_name, instructions[i - 1].argval))
        elif inst.opname == "STORE_NAME":
            names.setdefault(inst.argval, [])
            for prev in reversed(instructions[:i]):
                if prev.opname == "STORE_NAME":
                    break
                if prev.opname == "POP_TOP":
                    break
                if "JUMP" in prev.opname:
                    break
                if prev.opname == "IMPORT_FROM":
                    names[inst.argval].append(prev.argval)
                if prev.opname == "IMPORT_NAME":
                    names[inst.argval].append(prev.argval)
                    break
                if prev.opname == "LOAD_NAME":
                    names[inst.argval].append(prev.argval)
            names[inst.argval].reverse()
        elif (
            inst.opname in ["LOAD_ATTR", "LOAD_METHOD"]
            and instructions[i - 1].argval == "self"
        ):
            add_obj.add(".".join(obj_name.split(".")[:-1] + [inst.argval]))
        elif inst.opname == "LOAD_METHOD":
            methods[inst.argval] = instructions[i - 1].argval

    if "CITATION" in names:
        citations += names["CITATION"]
    for rm in get_recurse_methods(names, methods, add_obj=add_obj):
        citations = get_citations(*rm, done=done, citations=citations)
    return citations


def get_recurse_methods(name_dict, method_dict, add_obj=None):
    if add_obj is None:
        recurse_methods = set([])
    else:
        recurse_methods = add_obj
    for method, name_key in method_dict.items():
        if name_key not in name_dict:
            continue
        name_list = name_dict[name_key]
        tmp = []
        for name in name_list:
            if name in name_dict:
                for i in name_dict[name]:
                    if i in tmp:
                        continue
                    tmp.append(i)
            elif name not in tmp:
                tmp.append(name)
        recurse_methods.add(".".join(tmp + [method]))

    rv = set([])
    for rm in recurse_methods:
        module_name = rm
        obj = None
        while True:
            try:
                obj = importlib.import_module(module_name)
                break
            except ModuleNotFoundError:
                module_name = module_name.rsplit(".", maxsplit=1)[0]
        remainder = rm.replace(module_name + ".", "").split(".")
        if len(remainder) == 2:
            class_name, function_name = remainder[0], remainder[1]
        elif len(remainder) == 1:
            class_name, function_name = remainder[0], None
        elif len(remainder) == 0:
            class_name, function_name = None, None
        else:
            raise Exception

        all_classes = inspect.getmembers(obj, inspect.isclass)
        for c in all_classes:
            if c[0] == class_name:
                obj = c[1]
                break
        init_method = getattr(obj, "__init__")
        rv.add((init_method, "{}.{}.__init__".format(module_name, class_name)))
        if function_name is not None and hasattr(obj, function_name):
            obj = getattr(obj, function_name)
        rv.add((obj, rm))
    return rv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("-p", "--package", default="AaronTools")
    args = parser.parse_args()
    main(args)
