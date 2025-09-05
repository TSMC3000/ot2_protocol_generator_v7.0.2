import collections
import uuid
import os
from pathlib import Path
import itertools

# FIRST_TIP = 'A2'

OT2LabwareType = collections.namedtuple('OT2LabwareType', ['name', 'definition_id', 'max_vol', 'n_row', 'n_col', 'n_well'])
OT2LabwareItem = collections.namedtuple('OT2LabwareItem', ['display_name', 'slot', 'definition_id', 'uuid_path'])
OT2PipetteType = collections.namedtuple('OT2PipetteType', ['name', 'definition_id', 'min_vol', 'max_vol', 'tiprack_name'])

OT2PipetteTypeCollection = {
    'p300': OT2PipetteType('P300 Single-Channel GEN2', 'p300_single_gen2', 20, 300, 'OT_TIP_300uL'),
    'p1000': OT2PipetteType('P1000 Single-Channel GEN2', 'p1000_single_gen2', 100, 1000, 'OT_TIP_1000uL')
}

OT2LabwareTypeCollection = {
    'OT_TIP_1000uL': OT2LabwareType('Opentrons 96 Tip Rack 1000 µL', 'opentrons/opentrons_96_tiprack_1000ul/1', 1000, 8, 12, 96),
    'OT_TIP_300uL': OT2LabwareType('Opentrons 96 Tip Rack 300 µL', 'opentrons/opentrons_96_tiprack_300ul/1', 300, 8, 12, 96),
    'OT_TIP_20uL': OT2LabwareType('Opentrons 96 Tip Rack 20 µL', 'opentrons/opentrons_96_tiprack_20ul/1', 20, 8, 12, 96),
    'OT_TUBE_50ML': OT2LabwareType('Opentrons 6 Tube Rack with NEST 50 mL Conical', 'opentrons/opentrons_6_tuberack_nest_50ml_conical/1', 40000, 2, 3, 6),
    'OT_TUBE_50ML_15ML': OT2LabwareType('Opentrons 10 Tube Rack with NEST 4x50 mL, 6x15 mL Conical', 'opentrons/opentrons_10_tuberack_nest_4x50ml_6x15ml_conical/1', 40000, 3, 4, 10),
    'OT_TUBE_15ML': OT2LabwareType('Opentrons 15 Tube Rack with NEST 15 mL Conical', 'opentrons/opentrons_15_tuberack_nest_15ml_conical/1', 14000, 3, 5, 15),
    'OT_TUBE_1.5ML': OT2LabwareType('Opentrons 24 Tube Rack with NEST 1.5 mL Snapcap', 'opentrons/opentrons_24_tuberack_nest_1.5ml_snapcap/1', 1500, 4, 6, 24),
    'OT_TUBE_2ML': OT2LabwareType('Opentrons 24 Tube Rack with NEST 2 mL Snapcap', 'opentrons/opentrons_24_tuberack_nest_2ml_snapcap/1', 2000, 4, 6, 24),
    'OT_WELL_96w_360uL': OT2LabwareType('Corning 96 Well Plate 360 µL Flat', 'opentrons/corning_96_wellplate_360ul_flat/2', 360, 8, 12, 96),
    'OT_WELL_24w_3.4mL': OT2LabwareType('Corning 24 Well Plate 3.4 mL', 'opentrons/corning_24_wellplate_3.4ml_flat/2', 3400, 4, 6, 24),
}

LIB_OT2_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
ASSETS_DIR = LIB_OT2_DIR / 'assets'

class AESample:
    def __init__(self, sid, vol):
        self.sid = sid
        self.vol = vol

    def __str__(self):
        return f'{self.sid}, {self.vol}μL'

    def __repr__(self):
        return f'<AESample: {str(self)}>'


AETransferStep = collections.namedtuple('TransferStep', ['src_sid', 'src_slot', 'src_well', 'dst_sid', 'dst_slot', 'dst_well', 'vol', 'dst_tot_vol', 'pipette'])
AEPickUpTipStep = collections.namedtuple('PickUpTipStep', ['src_slot', 'src_well', 'pipette'])
AEDropTipStep = collections.namedtuple('DropTipStep', ['dst_slot', 'dst_well', 'pipette'])
AEMixingStep = collections.namedtuple('MixingStep', ['src_sid', 'src_slot', 'src_well', 'dst_tot_vol'])
AEPauseStep = collections.namedtuple('PauseStep', ['message', ])


MAIN_PIPETTE = 1000
ALT_PIPETTE = 300


def new_ot2_labware_item(display_name: str, slot: int, definition_id: str):
    if display_name == 'Trash':
        assert slot == 12
        definition_id = 'fixedTrash'
        uuid_path = 'fixedTrash'
    else:
        guid = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                              f'{definition_id.replace("/", "-")}.rack{slot}.ot2.matal.dev'))
        uuid_path = f'{guid}:{definition_id}'

    return OT2LabwareItem(
        display_name, slot, definition_id, uuid_path
    )


def gen_labware_wells_list(n_row: int, n_col: int, order='row_first'):
    if order == 'row_first':
        return ['%s%d' % w for w in itertools.product('ABCDEFGHIJKLMN'[:n_row], range(1, n_col + 1))]
    elif order == 'col_first':
        return ['%s%d' % (b, a) for a, b in itertools.product(range(1, n_col + 1), 'ABCDEFGHIJKLMN'[:n_row], )]

