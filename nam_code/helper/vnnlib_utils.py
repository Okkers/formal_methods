import os

NEGLIGIBLE_VALUE = 1e-8

def generate_vnnlib_property_full(input_value, epsilon, constant, eq, property_file, expl_indices):
    input_defs = ""
    input_constraints = ""
    output_defs = ""
    output_constraints = "(assert (or\n)"

    for i in range(len(input_value)):
        input_defs += f"(declare-const X_{i} Real)\n"
        if i in expl_indices:
            input_constraints += f"(assert (<= X_{i} {input_value[i] + 10**-30}))\n"
            input_constraints += f"(assert (>= X_{i} {input_value[i] - 10**-30}))\n"
        else:
            input_constraints += f"(assert (<= X_{i} {input_value[i] + epsilon}))\n"
            input_constraints += f"(assert (>= X_{i} {input_value[i] - epsilon}))\n"

    output_defs += f"(declare-const Y_0 Real)\n"

    if eq == "<":
        output_constraints = "(assert (or\n\t(and (<= Y_0 {0}))\n))\n".format(constant)
    else:
        assert eq == ">"
        output_constraints = "(assert (or\n\t(and (>= Y_0 {0}))\n))\n".format(constant + NEGLIGIBLE_VALUE)

    vnnlib_property = "; VNNLIB property for epsilon ball and winner\n\n"
    vnnlib_property += '; Definition of input variables\n' + input_defs + "\n"
    vnnlib_property += '; Definition of output variables\n' + output_defs + "\n"
    vnnlib_property += '; Definition of input constraints\n' + input_constraints + "\n"
    vnnlib_property += '; Definition of output constraints\n' + output_constraints + "\n"

    os.makedirs(os.path.dirname(property_file), exist_ok=True)
    with open(property_file, "w") as f:
        f.write(vnnlib_property)


def generate_vnnlib_property_partial(input_value, epsilon, constant, eq, property_file):
    input_defs = ""
    input_constraints = ""
    output_defs = ""
    output_constraints = "(assert (or\n)"

    input_defs += f"(declare-const X_0 Real)\n"
    input_constraints += f"(assert (<= X_0 {input_value + epsilon}))\n"
    input_constraints += f"(assert (>= X_0 {input_value - epsilon}))\n"

    output_defs += f"(declare-const Y_0 Real)\n"

    if eq == "<":
        output_constraints = "(assert (or\n\t(and (<= Y_0 {0}))\n))\n".format(constant)
    else:
        assert eq == ">"
        output_constraints = "(assert (or\n\t(and (>= Y_0 {0}))\n))\n".format(constant + NEGLIGIBLE_VALUE)

    vnnlib_property = "; VNNLIB property for epsilon ball and winner\n\n"
    vnnlib_property += '; Definition of input variables\n' + input_defs + "\n"
    vnnlib_property += '; Definition of output variables\n' + output_defs + "\n"
    vnnlib_property += '; Definition of input constraints\n' + input_constraints + "\n"
    vnnlib_property += '; Definition of output constraints\n' + output_constraints + "\n"

    os.makedirs(os.path.dirname(property_file), exist_ok=True)
    with open(property_file, "w") as f:
        f.write(vnnlib_property)