# from: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html

def large_num_to_reasonable_string(tick_val, decimals=1, bites=False):
    if abs(tick_val) >= 1e12:
        val = round(tick_val / 1e12, decimals)
        new_tick_format = '{:}T'.format(val)
    elif abs(tick_val) >= 1e9:
        val = round(tick_val / 1e9, decimals)
        if bites:
            new_tick_format = '{:}G'.format(val)
        else:
            new_tick_format = '{:}B'.format(val)
    elif abs(tick_val) >= 1e6:
        val = round(tick_val / 1e6, decimals)
        new_tick_format = '{:}M'.format(val)
    elif abs(tick_val) >= 1e3:
        val = round(tick_val / 1e3, decimals)
        new_tick_format = '{:}K'.format(val)
    elif abs(tick_val) < 1e3:
        new_tick_format = round(tick_val, decimals)
    else:
        new_tick_format = tick_val

    return new_tick_format


def reformat_large_tick_values(tick_val, pos):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    new_tick_format = large_num_to_reasonable_string(tick_val)

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)

    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2:]

    return new_tick_format


if __name__ == '__main__':
    a = large_num_to_reasonable_string(-1230000000, 2)
    print(a)
