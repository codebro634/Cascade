from typing import Union, Callable


def parse_bool(value: Union[bool,str]) -> bool:
    if isinstance(value,str):
        if value == "True":
            return True
        elif value == "False":
            return False
        else:
            raise ValueError(f"{value} is not a valid boolean value.")
    else:
        return value

def parse_list(value: Union[list,str], element_parser: Callable) -> list:
    if isinstance(value,str):
        value = value.strip()
        if value == "None":
            return None
        assert value.startswith('[') and value.endswith(']')
        value = value[1:-1]
        vals = [element_parser(split.strip()) for split in value.split(",") if split.strip()]
        return list(vals)
    else:
        return value

def parse_tuple(value: Union[tuple,str], element_parser: Callable) -> tuple:
    if isinstance(value,str):
        value = value.strip()
        if value == "None":
            return None
        assert value.startswith('(') and value.endswith(')')
        value = value[1:-1]
        vals = [element_parser(split.strip()) for split in value.split(",") if split.strip()]
        return tuple(vals)
    else:
        return value

