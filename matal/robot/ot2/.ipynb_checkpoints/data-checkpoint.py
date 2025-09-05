import collections
import itertools
import os
import uuid
from pathlib import Path

# FIRST_TIP = 'A2'

OT2LabwareType = collections.namedtuple('OT2LabwareType', ['name', 'definition_id', 'max_vol', 'n_row', 'n_col', 'n_well'])
OT2LabwareItem = collections.namedtuple('OT2LabwareItem', ['display_name', 'slot', 'definition_id', 'uuid_path'])

OT2LabwareTypeCollection = {
    'OT_TIP_1000uL': OT2LabwareType('Opentrons 96 Tip Rack 1000 µL', 'opentrons/opentrons_96_tiprack_1000ul/1', 1000, 8, 12, 96),
    'OT_TUBE_50ML': OT2LabwareType('6x50mL Tube Rack', 'opentrons/opentrons_6_tuberack_nest_50ml_conical/1', 40000, 2, 3, 6),
    'OT_TUBE_50ML_15ML': OT2LabwareType('4x50mL and 2x15mL Tube Rack', 'opentrons/opentrons_10_tuberack_nest_4x50ml_6x15ml_conical/1', 40000, 3, 4, 10),
    'OT_TUBE_15ML': OT2LabwareType('15x15mL Tube Rack', 'opentrons/opentrons_15_tuberack_nest_15ml_conical/1', 14000, 3, 5, 15),
    'CS_VIAL_15ML': OT2LabwareType('6x50mL Tube Rack', 'custom_beta/glassvialsincse_15_tuberack_15000ul/1', 14000, 3, 5, 15),
    # 'OT_TRASH': OT2_Labware_Type('Trash', 'opentrons/opentrons_1_trash_1100ml_fixed/1', 0, 0, 0, 0),
}


class AESample:
    def __init__(self, sid, vol):
        self.sid = sid
        self.vol = vol

    def __str__(self):
        return f'{self.sid}, {self.vol}μL'

    def __repr__(self):
        return f'<AESample: {str(self)}>'


AETransferStep = collections.namedtuple('TransferStep', ['src_sid', 'src_slot', 'src_well', 'dst_sid', 'dst_slot', 'dst_well', 'vol', 'dst_tot_vol'])
AEPickUpTipStep = collections.namedtuple('PickUpTipStep', ['src_slot', 'src_well'])
AEDropTipStep = collections.namedtuple('DropTipStep', ['dst_slot', 'dst_well'])
AEMixingStep = collections.namedtuple('MixingStep', ['src_sid', 'src_slot', 'src_well', 'dst_tot_vol'])


def new_ot2_labware_item(display_name: str, slot: int, definition_id: str):
    if display_name == 'Trash':
        assert slot == 12
        definition_id = 'opentrons/opentrons_1_trash_1100ml_fixed/1'
        uuid_path = 'trashId'
    else:
        guid = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                              f'{definition_id.replace("/", "-")}.rack{slot}.ot2.matal.dev'))
        uuid_path = f'{guid}:{definition_id}'

    return OT2LabwareItem(
        display_name, slot, definition_id, uuid_path
    )


def gen_labware_wells_list(n_row: int, n_col: int, n_wells: int = None):
    if n_wells:
        return ['%s%d' % w for w in itertools.product('ABCDEFGHIJKLMN'[:n_row], range(1, n_col + 1))][:n_wells]
    else:
        return ['%s%d' % w for w in itertools.product('ABCDEFGHIJKLMN'[:n_row], range(1, n_col + 1))]
    


LIB_OT2_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
ASSETS_DIR = LIB_OT2_DIR / 'assets'
