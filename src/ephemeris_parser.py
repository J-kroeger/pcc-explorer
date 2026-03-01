from datetime import datetime, timedelta

def _parse_rinex_float(s: str) -> float:
    """Converts a RINEX-formatted string to a float, handling 'D' for exponents."""
    s = s.strip()
    if not s or s == '':
        return 0.0
    # Replace 'D' with 'E' for scientific notation
    s = s.replace('D', 'E').replace('d', 'E')
    try:
        return float(s)
    except ValueError:
        print(f"Warning: Could not parse float from '{s}'")
        return 0.0

def read_ephemeris_records(record_lines: list) -> tuple:
    """
    Parses a list of 8 strings representing a single RINEX v2 or v3 ephemeris record.
    
    RINEX 2.x: Line 1 = PRN(3) + YY MM DD HH MM SS.S + 3 clock params
    RINEX 3.x: Line 1 = SYS+PRN(3) + YYYY MM DD HH MM SS + 3 clock params
    Data lines: v2 starts at col 3, v3 starts at col 4
    """
    params = {}
    
    # --- Line 1: PRN, Epoch and Clock Parameters ---
    line1 = record_lines[0]
    
    try:
        # Auto-detect RINEX version by checking if first char is a letter (v3) or digit/space (v2)
        first_char = line1[0]
        is_rinex3 = first_char.isalpha()
        
        if is_rinex3:
            # RINEX v3 format: "G01 2025 02 27 00 00 00  -3.862828E-04 ..."
            # Columns: 0-2 (SYS+PRN), 4-7 (YYYY), 8-10 (MM), 11-13 (DD), 
            #          14-16 (HH), 17-19 (MM), 20-22 (SS)
            year = int(line1[4:8].strip())
            month = int(line1[8:11].strip())
            day = int(line1[11:14].strip())
            hour = int(line1[14:17].strip())
            minute = int(line1[17:20].strip())
            second = float(line1[20:23].strip())
            epoch = datetime(year, month, day, hour, minute, int(second))
            
            # Clock parameters at columns 23, 42, 61
            params['SV_clock_bias'] = _parse_rinex_float(line1[23:42])
            params['SV_clock_drift'] = _parse_rinex_float(line1[42:61])
            params['SV_clock_drift_rate'] = _parse_rinex_float(line1[61:80])
            data_offset = 4  # RINEX v3 data lines start at column 4
        else:
            # RINEX v2 format: " 1 25  2 27  0  0  0.0  -3.862828E-04 ..."
            prn = int(line1[0:3].strip())
            year = int(line1[3:6].strip())
            month = int(line1[6:9].strip())
            day = int(line1[9:12].strip())
            hour = int(line1[12:15].strip())
            minute = int(line1[15:18].strip())
            second = float(line1[18:22].strip())
            
            if year >= 80:
                year += 1900
            else:
                year += 2000
            
            epoch = datetime(year, month, day, hour, minute, int(second))
            
            params['SV_clock_bias'] = _parse_rinex_float(line1[22:41])
            params['SV_clock_drift'] = _parse_rinex_float(line1[41:60])
            params['SV_clock_drift_rate'] = _parse_rinex_float(line1[60:79])
            data_offset = 3  # RINEX v2 data lines start at column 3
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing line 1: {e}")
        print(f"Line content: '{line1}'")
        raise
    
    # --- Lines 2-8: Orbital Parameters ---
    # Each line has 4 values of 19 chars each, starting at data_offset
    d = data_offset
    
    try:
        line2 = record_lines[1]
        params['IODE'] = _parse_rinex_float(line2[d:d+19])
        params['Crs'] = _parse_rinex_float(line2[d+19:d+38])
        params['Delta_n'] = _parse_rinex_float(line2[d+38:d+57])
        params['M0'] = _parse_rinex_float(line2[d+57:d+76])

        line3 = record_lines[2]
        params['Cuc'] = _parse_rinex_float(line3[d:d+19])
        params['e'] = _parse_rinex_float(line3[d+19:d+38])
        params['Cus'] = _parse_rinex_float(line3[d+38:d+57])
        params['sqrt_A'] = _parse_rinex_float(line3[d+57:d+76])

        line4 = record_lines[3]
        params['Toe'] = _parse_rinex_float(line4[d:d+19])
        params['Cic'] = _parse_rinex_float(line4[d+19:d+38])
        params['Omega0'] = _parse_rinex_float(line4[d+38:d+57])
        params['Cis'] = _parse_rinex_float(line4[d+57:d+76])

        line5 = record_lines[4]
        params['i0'] = _parse_rinex_float(line5[d:d+19])
        params['Crc'] = _parse_rinex_float(line5[d+19:d+38])
        params['omega'] = _parse_rinex_float(line5[d+38:d+57])
        params['Omega_dot'] = _parse_rinex_float(line5[d+57:d+76])

        line6 = record_lines[5]
        params['i_dot'] = _parse_rinex_float(line6[d:d+19])
        params['Codes_on_L2'] = _parse_rinex_float(line6[d+19:d+38])
        params['GPS_Week'] = _parse_rinex_float(line6[d+38:d+57])
        params['L2_P_Data_flag'] = _parse_rinex_float(line6[d+57:d+76])

        line7 = record_lines[6]
        params['SV_accuracy'] = _parse_rinex_float(line7[d:d+19])
        params['SV_health'] = _parse_rinex_float(line7[d+19:d+38])
        params['TGD'] = _parse_rinex_float(line7[d+38:d+57])
        params['IODC'] = _parse_rinex_float(line7[d+57:d+76])

        line8 = record_lines[7]
        params['Transmission_time'] = _parse_rinex_float(line8[d:d+19])
        
    except (ValueError, IndexError) as e:
        print(f"Error parsing orbital parameters: {e}")
        raise
    
    return epoch, params