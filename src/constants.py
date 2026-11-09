
from dataclasses import dataclass


@dataclass(frozen=True)
class SYSTEM:
    TXOP_DURATION: float = 5e-3    # seconds
    PN_DBM: int = -95              # dBm
    CCA: int = -82                 # dBm
    BW: int = 80                   # MHz
    NSS: int = 2                   # number of spatial streams
    
    TDFT: float = 12.8e-6               # OFDM DFT duration
    TGI: float = 0.8e-6                 # OFDM Guard Interval duration
    TIME_PREAMBLE_DATA: float = 100e-6   # Duration of the preamble for data frames
    TRTS: float = 56E-6                  # Duration of the RTS frame
    TCTS: float = 48E-6                  # Duration of the CTS frame
    TSIFS: float = 16e-6                 # Shortest Interframe spacing (SIFS time)
    DIFS: float = 34e-6                  # EDCA Interframe spacing (DIFS time)
    TE: float = 9e-6                     # Duration of a single backoff slot
    TBACK: float = 100E-6                # Block ACK duration

    TMAPC_ICF: float = 74.4E-6           # MAPC Initial Control Frame duration
    TMAPC_ICR: float = 88E-6             # MAPC Initial Control Response duration
    TMAPC_TF: float = 74.4E-6            # MAPC Trigger Frame duration

@dataclass(frozen=True)
class MAC:
    FRAME_LENGTH: int = 12e3       # bits
    LSF: int = 16                    # Length of service field (bits)
    LMH: int = 240                   # MAC header (bits)
    LTAIL: int = 18                  # Tail bits (bits)
    LMD: int = 32                    # MPDU Delimiter (bits)


@dataclass(frozen=True)
class CHANNEL:
    STD_DEV: float = 5.0        # dB
    FREQUENCY: float = 6.0    # GHz
    DBP: float = 10.0             # meters (breakpoint distance)
