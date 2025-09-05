import copy
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import uuid
from PyPDF2 import PdfMerger, PdfFileMerger

from matal.configs import OT2_PROC_DIR
from matal.utils import auto_log, get_sid_info, get_sid, Bunch, str_to_list
from .data_v7 import *
from .label_v7 import new_label_page


def quoted_presenter(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


yaml.add_representer(str, quoted_presenter)


def _compare_wells(w1, w2):
    if w1[0] != w2[0]:
        return -1 if w1[0] < w2[0] else 1
    else:
        w1_num, w2_num = int(w1[1:]), int(w2[1:])
        return -1 if w1_num < w2_num else 1


def split_slot_well(inp: str):
    """ '4A1' --> (4, 'A1'), '11A1' --> (11, A1) """
    if inp[1].isdigit():
        slot = inp[:2]
        well = inp[2:]
    else:
        slot = inp[0]
        well = inp[1:]
    assert slot.isdigit()
    assert well[0].isupper()
    assert well[1:].isdigit()

    return int(slot), well


class OT2Designer:
    def __init__(self, configs):
        self.ot2_proc = configs
        self.src_slots: list = str_to_list(self.ot2_proc.src_slots, func=int)
        self.dst_slots: list = str_to_list(self.ot2_proc.dst_slots, func=int)
        self.tip_slots: list = str_to_list(self.ot2_proc.tip_slots, func=int)
        self.alt_tip_slots: list = str_to_list(self.ot2_proc.alt_tip_slots, func=int)
        self.skipped_wells: list = str_to_list(self.ot2_proc.skipped_wells, func=split_slot_well)

        self.src_labware: OT2LabwareType = OT2LabwareTypeCollection[self.ot2_proc.src_labware_type]
        self.dst_labware: OT2LabwareType = OT2LabwareTypeCollection[self.ot2_proc.dst_labware_type]
        self.tip_labware: OT2LabwareType = OT2LabwareTypeCollection[self.ot2_proc.tip_labware_type]
        self.alt_tip_labware: OT2LabwareType = OT2LabwareTypeCollection[self.ot2_proc.alt_tip_labware_type]

        self.src_labware_wells: list = gen_labware_wells_list(self.src_labware.n_row, self.src_labware.n_col,
                                                              order=self.ot2_proc.src_well_order)
        self.dst_labware_wells: list = gen_labware_wells_list(self.dst_labware.n_row, self.dst_labware.n_col,
                                                              order=self.ot2_proc.dst_well_order)
        self.tip_labware_wells: list = gen_labware_wells_list(self.tip_labware.n_row, self.tip_labware.n_col)
        self.alt_tip_labware_wells: list = gen_labware_wells_list(self.alt_tip_labware.n_row,
                                                                  self.alt_tip_labware.n_col)

        if self.ot2_proc.separated_src_groups:
            self.separate_parts = True
            self.separated_src_groups = str_to_list(self.ot2_proc.separated_src_groups,
                                                    func=lambda s: tuple(
                                                        [i.strip() for i in s.split('+') if i.strip()]))
            self.seperated_src_map = {i + 1: v for i, v in enumerate(self.separated_src_groups)}
            self.seperated_src_reverse_map = {vi: k for k, v in self.seperated_src_map.items() for vi in v}
            self.n_parts = len(self.separated_src_groups)
        else:
            self.separate_parts = False
            self.seperated_src_map = None
            self.seperated_src_reverse_map = None
            self.n_parts = 1

        self.n_used_dst_wells = 0
        self.n_used_src_wells = 0
        self.src_wells = {}
        self.dst_wells = {}
        self.wells = {}
        self.init_wells_snapshot = {}
        self.transfers = []
        self.steps = []
        self.init_wells()
        self.tip_wells = []
        self.alt_tip_wells = []
        self.init_tips()

        labware_map = {12: new_ot2_labware_item('Trash', 12, '')}
        if self.ot2_proc.use_main_pipette:
            labware_map.update(
                {k: new_ot2_labware_item(f'tipRack{i + 1}', k, self.tip_labware.definition_id) for i, k in
                 enumerate(self.tip_slots)})

        if self.ot2_proc.use_alt_pipette:
            labware_map.update(
                {k: new_ot2_labware_item(f'altTipRack{i + 1}', k, self.alt_tip_labware.definition_id) for i, k in
                 enumerate(self.alt_tip_slots)})
        labware_map.update(
            {k: new_ot2_labware_item(f'srcRack{i + 1}', k, self.src_labware.definition_id) for i, k in
             enumerate(self.src_slots)})
        labware_map.update(
            {k: new_ot2_labware_item(f'dstRack{i + 1}', k, self.dst_labware.definition_id) for i, k in
             enumerate(self.dst_slots)})
        self.labware_map = labware_map

    def init_wells(self):
        # Fill slots with empty tubes
        for slot in self.src_slots:
            self.wells[slot] = {}
            for well in self.src_labware_wells:
                self.wells[slot][well] = AESample(None, 0)
        for slot in self.dst_slots:
            self.wells[slot] = {}
            for well in self.dst_labware_wells:
                self.wells[slot][well] = AESample(None, 0)

    def init_tips(self):
        # Find all available tips
        if self.ot2_proc.use_main_pipette:
            for slot in self.tip_slots:
                for well in self.tip_labware_wells:
                    if _compare_wells(well, self.ot2_proc.first_tip) < 0:
                        continue
                    self.tip_wells.append(tuple([slot, well]))

        if self.ot2_proc.use_alt_pipette:
            for slot in self.alt_tip_slots:
                for well in self.alt_tip_labware_wells:
                    if _compare_wells(well, self.ot2_proc.alt_first_tip) < 0:
                        continue
                    self.alt_tip_wells.append(tuple([slot, well]))

    def next_dst_slot(self):
        # Find the next empty dst well location
        while True:
            i = self.n_used_dst_wells
            self.n_used_dst_wells += 1

            try:
                slot_i = self.dst_slots[i // self.dst_labware.n_well]
            except IndexError:
                raise RuntimeError(f'Not enough dst_slots.')

            well_i = i % self.dst_labware.n_well
            well_s = self.dst_labware_wells[well_i]

            if (slot_i, well_s) not in self.skipped_wells:
                break
        return slot_i, well_s

    def next_src_slot(self):
        # Find the next empty src well location
        while True:
            i = self.n_used_src_wells
            self.n_used_src_wells += 1

            try:
                slot_i = self.src_slots[i // self.src_labware.n_well]
            except IndexError:
                raise RuntimeError(f'Not enough src_slots.')
            well_i = i % self.src_labware.n_well
            well_s = self.src_labware_wells[well_i]

            if (slot_i, well_s) not in self.skipped_wells:
                break

        return slot_i, well_s

    def add_src(self, sid, vol):
        # Add source solution and distribute them in available src well(s)
        max_vol_i = self.src_labware.max_vol - self.ot2_proc.src_extra_vol * 2
        while vol > 0:
            vol_i = max_vol_i if vol > max_vol_i else vol
            vol -= vol_i
            slot, well = self.next_src_slot()

            if sid in self.src_wells:
                self.src_wells[sid].append(tuple([slot, well]))
            else:
                self.src_wells[sid] = [tuple([slot, well]), ]

            self.wells[slot][well] = AESample(sid, vol_i + self.ot2_proc.src_extra_vol * 2)

    def initial_src_finished(self):
        self.init_wells_snapshot = copy.deepcopy(self.wells)

    def next_available_src_position(self, sid: str):
        # Find the first well that still have enough source for transferring
        all_pos = self.src_wells[sid]
        all_available_pos = []
        for slot, well in all_pos:
            vol_available = self.wells[slot][well].vol - self.ot2_proc.src_extra_vol
            if vol_available > 2 * self.ot2_proc.pipette_min_vol:
                all_available_pos.append((slot, well, vol_available))

        if len(all_available_pos) > 0:
            all_available_pos = sorted(all_available_pos, key=lambda c: c[2])
            return all_available_pos[-1]
        else:
            return None, None, 0

    def transfer_to_dst(self, src_sid, dst_slot, dst_well, vol):
        if self.ot2_proc.use_alt_pipette and (vol < self.ot2_proc.pipette_min_vol):
            return self._transfer_to_dst(src_sid, dst_slot, dst_well, vol,
                                         ALT_PIPETTE,
                                         self.ot2_proc.alt_pipette_max_vol,
                                         self.ot2_proc.alt_pipette_min_vol)
        else:
            return self._transfer_to_dst(src_sid, dst_slot, dst_well, vol,
                                         MAIN_PIPETTE,
                                         self.ot2_proc.pipette_max_vol,
                                         self.ot2_proc.pipette_min_vol)

    def _transfer_to_dst(self, src_sid, dst_slot, dst_well, vol, pipette, pipette_max_vol, pipette_min_vol):
        # Add source to a dst well, may result in multiple transfer steps
        dst_sid = self.wells[dst_slot][dst_well].sid
        while vol > 0:
            slot, well, vol_available = self.next_available_src_position(src_sid)

            vol_valid = min(vol_available, pipette_max_vol)
            vol_i = vol if vol <= vol_valid else vol_valid
            if 0 < vol - vol_i < pipette_min_vol:
                vol_i -= pipette_min_vol

            vol -= vol_i
            auto_log(f'{vol} {src_sid} --> {dst_slot}{dst_well}: {vol_i} from {slot}{well}', level='debug')
            self.wells[slot][well].vol -= vol_i
            self.wells[dst_slot][dst_well].vol += vol_i
            self.transfers.append(AETransferStep(src_sid, slot, well, dst_sid, dst_slot, dst_well, vol_i,
                                                 self.wells[dst_slot][dst_well].vol, pipette))
            if vol_i < pipette_min_vol:
                # Should never happen
                auto_log(f'Out-of-range value in AETransferStep: {src_sid}, {dst_slot}, {dst_well}, {vol}, {vol_i}',
                         level='error')

    def name_dst_well(self, slot, well, sid):
        self.wells[slot][well].sid = sid

    def generate_tip_steps(self, src_order=None):
        # Group transfers with the same src sID together and add PickUpTip & DropTip steps
        has_tips = {MAIN_PIPETTE: False, ALT_PIPETTE: False}
        tip_sids = {MAIN_PIPETTE: None, ALT_PIPETTE: None}
        
        if self.ot2_proc.randomize_transfers:
            rng = np.random.default_rng(int(self.ot2_proc.randomize_transfers))
            if src_order is None:
                transfers = sorted(self.transfers, key=lambda t: (t.src_sid, rng.random()))
            else:
                transfers = sorted(self.transfers, key=lambda t: (src_order.index(t.src_sid), rng.random())) 
        else:
            
            if src_order is None:
                transfers = sorted(self.transfers, key=lambda t: t.src_sid)
            else:
                transfers = sorted(self.transfers, key=lambda t: src_order.index(t.src_sid))

        num_tips_used = {MAIN_PIPETTE: 0, ALT_PIPETTE: 0}
        for transfer in transfers:
            src_sid = transfer.src_sid
            if tip_sids[transfer.pipette] != src_sid:
                if has_tips[transfer.pipette]:
                    self.steps.append(AEDropTipStep(12, 'A1', transfer.pipette))
                    if src_sid in self.ot2_proc.pause_after_src:
                        self.steps.append(AEPauseStep(f'After transfering {src_sid}'))

                if transfer.pipette == MAIN_PIPETTE:
                    slot, well = self.tip_wells.pop(0)
                elif transfer.pipette == ALT_PIPETTE:
                    slot, well = self.alt_tip_wells.pop(0)
                else:
                    raise ValueError(f'Unknown pipette: {transfer.pipette}')

                if src_sid in self.ot2_proc.pause_before_src:
                    self.steps.append(AEPauseStep(f'Before transfering {src_sid}'))
                self.steps.append(AEPickUpTipStep(slot, well, transfer.pipette))
                num_tips_used[transfer.pipette] += 1

                tip_sids[transfer.pipette] = src_sid
                has_tips[transfer.pipette] = True
            self.steps.append(transfer)
        self.steps.append(AEDropTipStep(12, 'A1', MAIN_PIPETTE))
        if src_sid in self.ot2_proc.pause_after_src:
            self.steps.append(AEPauseStep(f'After transfering {src_sid}'))
        # auto_log(f'Number of used tips: {num_tips_used}, next tip location: {self.tip_wells[0]}')

    def insert_mixing_steps(self, all_src_sid: list):
        if self.ot2_proc.use_alt_pipette is False:
            raise NotImplementedError('TODO: mixing with alt pipette')
        mix_vol = self.ot2_proc.pipette_max_vol
        after_src = self.ot2_proc.mixing_after_src
        repeats = self.ot2_proc.mixing_rounds

        mixing_transfers = []
        dst_sids = []
        for slot in self.wells.keys():
            for well in self.wells[slot].keys():
                sid: str = self.wells[slot][well].sid
                vol: int = self.wells[slot][well].vol
                if sid and sid[0].islower() and vol > 0:
                    dst_sids.append(sid)
                    mixing_transfers.extend([
                        AETransferStep(sid, slot, well, sid, slot, well, mix_vol, vol, MAIN_PIPETTE) for _ in
                        range(repeats)])

        self.transfers.extend(mixing_transfers)

        if not after_src:
            return all_src_sid + dst_sids
        else:
            idx = all_src_sid.index(after_src) + 1
            return all_src_sid[:idx] + dst_sids + all_src_sid[idx:]


def get_default_configs():
    c = dict(
        proj_code=None,
        proj_owner=None,
        batch_sid=None,
        batch_desc=None,
        proc_sid=None,
        proc_sid_start=0,
        proc_sid_i=1,
        sample_file=None,
        add_water=False,
        water_sid='WTR-0000',
        water_target_vol=0,
        water_target_mass_loading=None,
        frc_one=10000,
        n_sources_per_sample=None,
        sources=dict(),
        src_order=None,
        water_first=True,
        scaling_factor=1,
        n_replicates=0,
        start_idx=0,
        end_idx=-1,
        randomize=False,
        randomize_transfers=False,
        n_samples_per_proc=0,
        src_labware_type='OT_TUBE_50ML',
        dst_labware_type='OT_TUBE_15ML',
        tip_labware_type='OT_TIP_1000uL',
        alt_tip_labware_type='OT_TIP_300uL',
        src_slots='2,3,6,9',
        src_well_order='row_first',
        skip_srcs=list(),
        dst_slots='4,5,7,8',
        dst_well_order='row_first',
        alt_tip_slots='2',
        alt_first_tip='A2',
        tip_slots='1',
        first_tip='A2',
        separated_src_groups='',
        skipped_wells='',
        src_extra_vol=5000,
        use_alt_pipette=False,
        use_main_pipette=True,
        alt_pipette_max_vol=260,
        alt_pipette_min_vol=20,
        alt_air_gap_vol=20,
        pipette_max_vol=800,
        pipette_min_vol=100,
        air_gap_vol=100,
        aspirate_rate=137.35,
        alt_aspirate_rate=137.35 / 5,
        alt_aspirate_rate_srcs='',
        aspirate_delay=0.0,
        airgap_delay=0.0,
        dispense_delay=0.0,
        blowout_delay=0.0,
        skip_delay_srcs=list(),
        dispense_rate=137.35,
        alt_dispense_rate=137.35 / 5,
        alt_dispense_rate_srcs='',
        aspirate_offset=5,
        air_gap_offset=120,
        dispense_offset_a=0.007692307692307693,
        # dispense_offset_b=20,
        # dispense_offset_max=125,
        # Temporary mitigation for issue #1
        dispense_offset_b=110,
        dispense_offset_max=120,
        mixing=False,
        mixing_rounds=3,
        mixing_after_src=None,
        src_mixing_rounds=0,
        src_mixing_vol=-1,
        src_mixing_rate=-1,
        src_mixing_delay=0,
        src_mixing_offset=-1,
        skip_src_mixing_srcs=list(),
        label_template='avery_5267',
        label_types='ot2_src,ot2_dst,dish_comp',
        label_grouping='by_type',
        label_break_between_groups='none',
        label_filename=None,
        proc_filename=None,
        pause_after_src=list(),
        pause_before_src=list(),
    )
    return c


class OT2ProcGenerator:
    def __init__(self, sample_df: Union[str, pd.DataFrame], config_fmt: str = 'yaml',
                 in_dir: str = None, out_dir: str = None, configs: dict = None, unsafe_config=False):
        self.configs = get_default_configs()

        if isinstance(sample_df, str):
            warnings.warn('Init OT2ProcGenerator() with input_name is deprecated. '
                          'Please use with OT2ProcGenerator(sample_df, config=config) instead.',
                          DeprecationWarning)
            raise NotImplementedError('Init OT2ProcGenerator() with input_name is deprecated. ')
        else:
            self.configs.update(configs)
            self.configs = Bunch(**self.configs)
            self.samples = sample_df.copy(deep=True)
            self.input_name = 'DataFrame'

        if (self.configs.proj_code is None) and (isinstance(self.configs.batch_sid, str)) and (
                len(self.configs.batch_sid) > 2):
            self.configs.proj_code = self.configs.batch_sid[0].lower() + self.configs.batch_sid[1].upper()

        try:
            self._check_configs()
        except Exception as e:
            if not unsafe_config:
                raise (e)

        if self.configs.n_sources_per_sample is None:
            for n in range(1, 100):
                if f'src{n}_sid' not in self.samples.columns:
                    self.configs.n_sources_per_sample = n - 1
                    break
            else:
                auto_log('Auto-detection of n_sources_per_sample stopped at 100 and failed.')
            auto_log(f'Auto-detection of n_sources_per_sample = {self.configs.n_sources_per_sample}.')

        self.n_samples = len(self.samples)

        self.out_dir = out_dir
        self.proc_sids = []

    def _check_configs(self):
        assert isinstance(self.configs.proj_code, str), 'proj_code must be a string'
        assert len(self.configs.proj_code) == 2, 'length of proj_code must be 2'
        assert self.configs.proj_code[0].islower(), 'first character of proj_code must be a lowercase letter'
        assert self.configs.proj_code[1].isupper(), 'second character of proj_code must be a uppercase letter'

        assert isinstance(self.configs.proj_owner, str), 'proj_owner must be a string'
        assert len(self.configs.proj_owner) == 2, 'length of proj_owner must be 2'
        assert self.configs.proj_owner.isupper(), 'proj_owner must be in uppercase'

        assert isinstance(self.configs.batch_sid, str), 'batch_sid must be a string'
        assert self.configs.batch_sid.startswith(self.configs.proj_code), 'batch_sid must start with proj_code'
        assert get_sid_info(self.configs.batch_sid)['item_type'] == 'Batch', 'item_type of batch_sid must be "Batch"'

        assert isinstance(self.configs.batch_desc, str), 'batch_desc must be a string'
        assert len(self.configs.batch_desc) > 0, 'length of batch_desc must be greater than 0'

        if self.configs.proc_sid:
            assert isinstance(self.configs.proc_sid, str), 'proc_sid must be a string'
            assert self.configs.proc_sid.startswith(self.configs.proj_code), 'proc_sid must start with proj_code'
            assert get_sid_info(self.configs.proc_sid)[
                       'item_type'] == 'OT2Protocol', 'item_type of proc_sid must be "OT2Protocol"'

        if self.configs.proc_sid_start:
            assert isinstance(self.configs.proc_sid_start, int), 'proc_sid_start must be an integer'
            assert self.configs.proc_sid_start >= 0, 'proc_sid_start must be equal or greater than 0'
        if self.configs.proc_sid_i:
            assert isinstance(self.configs.proc_sid_i, int), 'proc_sid_i must be an integer'
            assert self.configs.proc_sid_i > 0, 'proc_sid_i must be greater than 0'

        # TODO: check for configs.sample_file

        assert isinstance(self.configs.add_water, bool), 'add_water must be a boolean'

        assert isinstance(self.configs.water_sid, str), 'water_sid must be a string'
        assert get_sid_info(self.configs.water_sid)['item_type'] == 'Source', 'item_type of water_sid must be "Source"'

        assert isinstance(self.configs.water_target_vol, int), 'water_target_vol must be an integer'
        assert self.configs.water_target_vol >= 0, 'water_target_vol must be equal or greater than 0'

        if isinstance(self.configs.n_sources_per_sample, int):
            assert self.configs.n_sources_per_sample > 0, 'n_sources_per_sample must be greater than 0'
        else:
            assert self.configs.n_sources_per_sample is None, 'n_sources_per_sample must be None if not an integer'

        assert isinstance(self.configs.frc_one, int), 'frc_one must be an integer'
        assert self.configs.frc_one > 0, 'frc_one must be greater than 0'

        assert isinstance(self.configs.sources, dict), 'sources must be a dictionary'
        for src_sid, src_dict in self.configs.sources.items():
            assert isinstance(src_sid, str), 'Each key of sources must be a string'
            assert get_sid_info(src_sid)['item_type'] == 'Source', 'item_type of each key in sources must be "Source"'

            assert isinstance(src_dict, dict), 'Each value of sources must be a dictionary'
            assert 'concentration' in src_dict, 'Each value of sources must have a key "concentration"'
            assert isinstance(src_dict['concentration'], str), 'Each concentration of sources must be a string'
            assert len(src_dict['concentration']) > 0, 'length of each concentration of sources must be greater than 0'

        assert isinstance(self.configs.scaling_factor, (int, float)), 'scaling_factor must be an integer or float'
        assert self.configs.scaling_factor > 0, 'scaling_factor must be greater than 0'

        assert isinstance(self.configs.n_replicates, int), 'n_replicates must be an integer'
        assert self.configs.n_replicates >= 0, 'n_replicates must be equal or greater than 0'

        assert isinstance(self.configs.start_idx, int), 'start_idx must be an integer'
        assert self.configs.start_idx >= 0, 'start_idx must be equal or greater than 0'

        assert isinstance(self.configs.end_idx, int), 'end_idx must be an integer'
        assert self.configs.end_idx == -1 or self.configs.end_idx > self.configs.start_idx, 'end_idx must be -1 or greater than start_idx'

        assert isinstance(self.configs.randomize, (bool, int)), 'randomize must be a boolean or integer'

        assert isinstance(self.configs.n_samples_per_proc, int), 'n_samples_per_proc must be an integer'
        assert self.configs.n_samples_per_proc >= 0, 'n_samples_per_proc must be equal or greater than 0'

        assert isinstance(self.configs.src_labware_type, str), 'src_labware_type must be a string'
        assert self.configs.src_labware_type in OT2LabwareTypeCollection, 'src_labware_type must be in OT2LabwareTypeCollection'

        assert isinstance(self.configs.dst_labware_type, str), 'dst_labware_type must be a string'
        assert self.configs.dst_labware_type in OT2LabwareTypeCollection, 'dst_labware_type must be in OT2LabwareTypeCollection'

        assert isinstance(self.configs.tip_labware_type, str), 'tip_labware_type must be a string'
        assert self.configs.tip_labware_type in OT2LabwareTypeCollection, 'tip_labware_type must be in OT2LabwareTypeCollection'

        assert isinstance(self.configs.src_slots, str), 'src_slots must be a string'
        src_slots = str_to_list(self.configs.src_slots, func=int)
        assert all([1 <= s <= 11 for s in src_slots]), 'all src_slots must be within range 1 to 11'

        assert isinstance(self.configs.dst_slots, str), 'dst_slots must be a string'
        dst_slots = str_to_list(self.configs.dst_slots, func=int)
        assert all([1 <= s <= 11 for s in dst_slots]), 'all dst_slots must be within range 1 to 11'

        assert isinstance(self.configs.tip_slots, str), 'tip_slots must be a string'
        tip_slots = str_to_list(self.configs.tip_slots, func=int)
        assert all([1 <= s <= 11 for s in tip_slots]), 'all tip_slots must be within range 1 to 11'

        assert len(tip_slots) == 1, 'length of tip_slots must be 1'
        assert len(set(src_slots + dst_slots + tip_slots)) == len(
            src_slots + dst_slots + tip_slots), 'All slots must be unique'

        assert isinstance(self.configs.first_tip, str), 'first_tip must be a string'
        assert self.configs.first_tip[0].isupper(), 'first character of first_tip must be a uppercase letter'
        assert self.configs.first_tip[1:].isdigit(), 'remaining characters of first_tip must be digits'

        assert isinstance(self.configs.separated_src_groups, str), 'separated_src_groups must be a string'
        separated_src_groups = str_to_list(self.configs.separated_src_groups)
        for group in separated_src_groups:
            sids = str_to_list(group, sep='+')
            assert all([get_sid_info(sid)['item_type'] == 'Source' for sid in
                        sids]), 'all sids in separated_src_groups must be "Source" item_type'

        assert isinstance(self.configs.skipped_wells, str), 'skipped_wells must be a string'
        str_to_list(self.configs.skipped_wells, func=split_slot_well)

        assert isinstance(self.configs.src_extra_vol, int), 'src_extra_vol must be an integer'
        assert 0 <= self.configs.src_extra_vol, 'src_extra_vol must be equal or greater than 0'
        assert self.configs.src_extra_vol < OT2LabwareTypeCollection[
            self.configs.src_labware_type].max_vol, 'src_extra_vol must be less than max_vol of src_labware_type'

        tip_max_vol = OT2LabwareTypeCollection[self.configs.tip_labware_type].max_vol
        assert isinstance(self.configs.pipette_max_vol, int), 'pipette_max_vol must be an integer'
        assert 0 < self.configs.pipette_max_vol, 'pipette_max_vol must be greater than 0'
        assert self.configs.pipette_max_vol < tip_max_vol, 'pipette_max_vol must be less than max_vol of tip_labware_type'

        assert isinstance(self.configs.pipette_min_vol, int), 'pipette_min_vol must be an integer'
        assert 0 < self.configs.pipette_min_vol, 'pipette_min_vol must be greater than 0'
        assert self.configs.pipette_min_vol < tip_max_vol, 'pipette_min_vol must be less than max_vol of tip_labware_type'

        assert isinstance(self.configs.air_gap_vol, int), 'air_gap_vol must be an integer'
        assert 0 <= self.configs.air_gap_vol, 'air_gap_vol must be equal or greater than 0'
        assert self.configs.pipette_max_vol < tip_max_vol, 'pipette_max_vol must be less than max_vol of tip_labware_type'

        assert self.configs.pipette_min_vol < self.configs.pipette_max_vol, 'pipette_min_vol must be less than pipette_max_vol'
        assert self.configs.pipette_max_vol < (
                tip_max_vol - self.configs.air_gap_vol), 'pipette_max_vol must be less than max_vol of tip_labware_type minus air_gap_vol'
        assert self.configs.pipette_min_vol < (
                tip_max_vol - self.configs.air_gap_vol), 'pipette_min_vol must be less than max_vol of tip_labware_type minus air_gap_vol'

        assert isinstance(self.configs.aspirate_rate, (int, float)), 'aspirate_rate must be an integer or float'
        assert self.configs.aspirate_rate > 0, 'aspirate_rate must be greater than 0'

        assert isinstance(self.configs.dispense_rate, (int, float)), 'dispense_rate must be an integer or float'
        assert self.configs.dispense_rate > 0, 'dispense_rate must be greater than 0'

        assert isinstance(self.configs.aspirate_offset, (int, float)), 'aspirate_offset must be an integer or float'
        assert self.configs.aspirate_offset > 0, 'aspirate_offset must be greater than 0'

        assert isinstance(self.configs.air_gap_offset, (int, float)), 'air_gap_offset must be an integer or float'
        assert self.configs.air_gap_offset > 0, 'air_gap_offset must be greater than 0'

        assert isinstance(self.configs.dispense_offset_a, (int, float)), 'dispense_offset_a must be an integer or float'
        assert self.configs.dispense_offset_a > 0, 'dispense_offset_a must be greater than 0'

        assert isinstance(self.configs.dispense_offset_b, (int, float)), 'dispense_offset_b must be an integer or float'
        assert self.configs.dispense_offset_b > 0, 'dispense_offset_b must be greater than 0'

        assert isinstance(self.configs.dispense_offset_max,
                          (int, float)), 'dispense_offset_max must be an integer or float'
        assert self.configs.dispense_offset_max > 0, 'dispense_offset_max must be greater than 0'

        assert isinstance(self.configs.mixing, bool), 'mixing must be a boolean'

        assert isinstance(self.configs.mixing_rounds, int), 'mixing_rounds must be an integer'
        assert self.configs.mixing_rounds > 0, 'mixing_rounds must be greater than 0'

        assert (self.configs.mixing_after_src is None) or isinstance(self.configs.mixing_after_src,
                                                                     str), 'mixing_after_src must be None or a string'
        if self.configs.mixing_after_src is not None:
            assert get_sid_info(self.configs.mixing_after_src)[
                       'item_type'] == 'Source', 'item_type of mixing_after_src must be "Source" if not None'

        assert isinstance(self.configs.label_template, str), 'label_template must be a string'
        assert isinstance(self.configs.label_types, str), 'label_types must be a string'

        assert isinstance(self.configs.label_grouping, str), 'label_grouping must be a string'
        assert self.configs.label_grouping in ['none', 'by_type',
                                               'by_slot'], 'label_grouping must be either "by_type" or "by_slot"'

        assert isinstance(self.configs.label_break_between_groups, str), 'label_break_between_groups must be a string'
        assert self.configs.label_break_between_groups in ['none', 'page',
                                                           'column'], 'label_break_between_groups must be either "none", "page", or "column"'

        assert self.configs.label_filename is None or isinstance(self.configs.label_filename,
                                                                 str), 'label_filename must be None or a string'

    def generate(self, out_dir: str = None):
        if out_dir is not None:
            self.out_dir = out_dir

        if self.configs.separated_src_groups:
            n_parts = len(str_to_list(self.configs.separated_src_groups))
        else:
            n_parts = 1

        n_prepared = getattr(self.configs, 'start_idx', 0)
        end = getattr(self.configs, 'end_idx', None)
        n_needed = end if end and end > 0 else self.n_samples
        n_replicates = getattr(self.configs, 'n_replicates', 0)
        randomize = getattr(self.configs, 'randomize', False)
        i = self.configs.proc_sid_i

        n_samples_per_proc = getattr(self.configs, 'n_samples_per_proc', None)
        if not n_samples_per_proc or n_samples_per_proc <= 0:
            n_dst_wells_per_rack = OT2LabwareTypeCollection[self.configs.dst_labware_type].n_well
            n_dst_racks = len(str_to_list(self.configs.dst_slots))
            n_dst_wells = n_dst_wells_per_rack * n_dst_racks
            skipped_dst_wells = [(s, w) for s, w in str_to_list(self.configs.skipped_wells, func=split_slot_well) if
                                 s in str_to_list(self.configs.dst_slots, func=int)]
            n_skipped_dst_wells = len(skipped_dst_wells)
            n_samples_per_proc = (n_dst_wells - n_skipped_dst_wells) // n_parts // (1 + n_replicates)
        n_proc = math.ceil(self.n_samples / n_samples_per_proc)

        while n_prepared < n_needed:
            if not self.configs.proc_sid:
                proc_sid = get_sid('OT2Protocol', self.configs.batch_sid, i=i + self.configs.proc_sid_start)
            else:
                proc_sid = self.configs.proc_sid
            self.proc_sids.append(proc_sid)

            start_idx = n_prepared
            end_idx = min(n_prepared + n_samples_per_proc, n_needed)

            desc = f'OT2 Proc #{i}/{n_proc} for batch {self.configs.batch_sid}: {end_idx - start_idx} samples.'
            auto_log(f'input_name={self.input_name}, i={i}, '
                     f'n_samples={self.n_samples}, n_needed={n_needed}, n_prepared={n_prepared}, '
                     f'n_samples_per_proc={n_samples_per_proc}, start_idx={start_idx}, '
                     f'end_idx={end_idx}, ot2_proc_sid={proc_sid}')

            proc_configs = copy.deepcopy(self.configs)
            proc_configs.proc_sid = proc_sid
            proc_configs.proc_sid_i = i
            proc_configs.sample_file = os.path.basename(self.get_asset_path(proc_sid, 'debug_samples'))
            proc_configs.start_idx = start_idx
            proc_configs.end_idx = end_idx
            with open(self.get_asset_path(proc_sid, 'debug_configs'), 'w') as f:
                yaml.dump(dict(proc_configs), f, sort_keys=False, default_flow_style=False)
            self.samples.to_csv(self.get_asset_path(proc_sid, 'debug_samples'), index=False)

            self._gen_ot2_well_df(proc_sid, start=start_idx, end=end_idx, randomize=randomize,
                                  n_replicates=n_replicates)
            designer = self._gen_ot2_design(proc_sid)

            self._gen_ot2_json(proc_sid, designer, desc=desc)
            self._gen_ot2_sources(proc_sid, designer)
            self._gen_labels(proc_sid)

            n_prepared += end_idx - start_idx
            i += 1
        return copy.deepcopy(self.proc_sids)

    def _gen_ot2_well_df(self, proc_sid: str, start=0, end=-1, randomize=False, n_replicates=0):
        """
        Generate an OT-2 well list. Start and end can be used to specify the range of samples in the batch.
        if *randomize* is set to True, then the list of samples will be shuffled before applying the range.
        This is useful if you have a large batch and need multiple OT2Protocols to finish that batch, but you want
        to randomize sample placements among those OT2Protocols. Since the randomize parameter will also be used
        as the random seed of the shuffle operation, all samples in the batch will be included in one of those
        OT2Protocols exactly once.
        """
        samples = [s for _, s in self.samples.iterrows()]
        if randomize:
            r = random.Random(int(randomize))
            r.shuffle(samples)
            samples = samples[start:end]
            samples = sorted(samples, key=lambda s: s.sample_sid)
        else:
            samples = samples[start:end]
        columns = ['sample_sid', 'batch_sid', 'batch_i', 'proc_i', 'part_i', 'n_parts', 'ot2_slot', 'ot2_well']
        for i in range(1, self.configs.n_sources_per_sample + 1):
            columns.append(f'src{i}_sid')
            columns.append(f'src{i}_frc')
            columns.append(f'src{i}_vol')
            columns.append(f'src{i}_part_vol')
        columns.extend(['water_vol', 'water_part_vol', 'part_vol', 'total_vol', 'total_mass', 'additional_flags'])
        df = pd.DataFrame(columns=columns)
        df = df.astype({'additional_flags': str})

        design = OT2Designer(self.configs)

        for part_i in range(1, design.n_parts + 1):
            for _ in range(n_replicates + 1):
                for i, sample in enumerate(samples):
                    try:
                        sample_info = get_sid_info(sample.sample_sid)
                        sample_idx = sample_info['idx_int']
                        if sample_info['batch_sid'] != self.configs.batch_sid:
                            auto_log(f'Sample sID {sample.sample_sid} does not belong to batch sid'
                                     f' {self.configs.batch_sid}', level='warning')
                            sample_idx = -1
                    except (KeyError, ValueError) as e:
                        auto_log(f'Can not decode sample_sid {sample.sample_sid}: {e}', level='warning')
                        sample_idx = -1

                    slot, well = design.next_dst_slot()
                    line = [sample.sample_sid, self.configs.batch_sid,
                            sample_idx, i + 1, part_i, design.n_parts,
                            slot, well]

                    tot_vol = 0
                    part_vol = 0
                    for src_i in range(1, self.configs.n_sources_per_sample + 1):
                        src_sid = getattr(sample, f'src{src_i}_sid')
                        src_vol = round(getattr(sample, f'src{src_i}_vol') * self.configs.scaling_factor)
                        line.append(src_sid)
                        line.append(getattr(sample, f'src{src_i}_frc', 0) / self.configs.frc_one)
                        line.append(src_vol)
                        if (not design.separate_parts) or (src_sid in design.seperated_src_map[part_i]):
                            line.append(src_vol)
                            part_vol += src_vol
                        else:
                            line.append(0)
                            part_vol += 0
                        tot_vol += src_vol

                    water_target_vol = OT2LabwareTypeCollection[self.configs.dst_labware_type].max_vol
                    if self.configs.water_target_vol is None:
                        if self.configs.water_target_mass_loading is not None:
                            water_target_vol = round(
                                sample.mass * self.configs.scaling_factor / self.configs.water_target_mass_loading)
                    elif self.configs.water_target_vol > 0:
                        water_target_vol = self.configs.water_target_vol
                    else:
                        pass

                    water_vol = water_target_vol - tot_vol if self.configs.add_water else 0
                    # water_vol = 0 if (water_vol < self.configs.pipette_min_vol) and (
                            # self.configs.use_alt_pipette is False) else water_vol
                    water_vol = 0 if (water_vol < self.configs.pipette_min_vol) else water_vol
                    line.append(water_vol)
                    if (not design.separate_parts) or (self.configs.water_sid in design.seperated_src_map[part_i]):
                        line.append(water_vol)
                        part_vol += water_vol
                    else:
                        line.append(0)
                        part_vol += 0
                    line.append(part_vol)
                    line.append(tot_vol + water_vol)
                    line.append(sample.mass * self.configs.scaling_factor)
                    additional_flags = sample.additional_flags

                    if isinstance(additional_flags, (np.floating, float)) and np.isnan(additional_flags):
                        additional_flags = ''
                    else:
                        if not isinstance(additional_flags, str):
                            additional_flags = str(additional_flags)
                    line.append(', '.join(str_to_list(additional_flags)))
                    df.loc[len(df)] = line

        df.to_csv(self.get_asset_path(proc_sid, 'debug_ot2_wells'), index=False, float_format='%.4f')

    def _gen_ot2_design(self, proc_sid: str):
        """
            Generate an OT2Designer object based on the given OT2Protocol, _gen_ot2_df() must be called before this
            to make sure the 'debug_ot2_wells' asset is generated. The returned OT2Designer object has already
            included all source placements and all transfer steps.
            """

        # Load batch information
        well_df = pd.read_csv(self.get_asset_path(proc_sid, 'debug_ot2_wells'),
                              dtype={'additional_flags': str}, keep_default_na=False)
        designer = OT2Designer(self.configs)

        # Extract all types of source sIDs from the batch DataFrame
        seq = []
        for i in range(1, self.configs.n_sources_per_sample + 1):
            seq.extend(list(well_df[f'src{i}_sid'].values))
        seen = set()
        seen_add = seen.add
        all_src_sid = [x for x in seq if not (x in seen or seen_add(x))]

        # Calculate the total volume of sources needed
        all_src_vol = {sid: 0 for sid in all_src_sid}
        all_src_vol[self.configs.water_sid] = 0
        for _, row in well_df.iterrows():
            for i in range(1, self.configs.n_sources_per_sample + 1):
                all_src_vol[row[f'src{i}_sid']] += row[f'src{i}_part_vol']
            all_src_vol[self.configs.water_sid] += row['water_part_vol']

        # Place source tubes on OT2 racks
        for sid, vol in all_src_vol.items():
            if sid not in self.configs.skip_srcs:
                designer.add_src(sid, vol)

        # Take a snapshot of the initial source placements
        designer.initial_src_finished()

        # Add transfer steps for destination tubes
        for _, row in well_df.iterrows():
            sid, dst_slot, dst_well = row['sample_sid'], row['ot2_slot'], row['ot2_well']
            designer.name_dst_well(dst_slot, dst_well, sid)
            designer.transfer_to_dst(self.configs.water_sid, dst_slot, dst_well, row['water_part_vol'])

            for i in range(1, self.configs.n_sources_per_sample + 1):
                src_sid, src_vol = row[f'src{i}_sid'], row[f'src{i}_part_vol']
                if src_sid not in self.configs.skip_srcs:
                    designer.transfer_to_dst(src_sid, dst_slot, dst_well, src_vol)

        if self.configs.src_order is not None:
            src_order = self.configs.src_order + [s for s in all_src_sid if s not in self.configs.src_order]
        else:
            src_order = all_src_sid

        # Append water sID
        if self.configs.add_water:
            if self.configs.water_first:
                src_order = [self.configs.water_sid] + src_order
            else:
                src_order = src_order + [self.configs.water_sid]

        # Append sample sIDs
        if self.configs.mixing:
            src_order = designer.insert_mixing_steps(src_order)

        # Generate tip steps (PickUpTip & DropTip), the designer will group up transfer steps based on the
        # aspirate sID to save tip usages while not contaminating source solutions.
        designer.generate_tip_steps(src_order=src_order)

        return designer

    def _need_delay(self, command, src_sid):
        if src_sid == 'WTR-0000':
            pass
        elif command == 'aspirate':
            if (self.configs.aspirate_delay > 0) and (src_sid not in self.configs.skip_delay_srcs):
                return True
        elif command == 'airgap':
            if (self.configs.airgap_delay > 0) and (src_sid not in self.configs.skip_delay_srcs):
                return True
        elif command == 'dispense':
            if (self.configs.dispense_delay > 0) and (src_sid not in self.configs.skip_delay_srcs):
                return True
        elif command == 'blowout':
            if (self.configs.blowout_delay > 0) and (src_sid not in self.configs.skip_delay_srcs):
                return True
        return False

    def _get_command_uuid(self):
        return str(uuid.uuid4())

    def _gen_ot2_json(self, proc_sid: str, designer: OT2Designer, desc: str = None):
        with open(ASSETS_DIR / 'template_v7.json') as f:
            proc = json.loads(f.read())

        na = np.nan

        all_src_sid = designer.src_wells.keys()

        colors = ['#1f1fb4',
                  '#ffff0e',
                  '#2c2c2c',
                  '#d6d628',
                  '#9494bd',
                  '#8c8c4b',
                  '#e3e3c2',
                  '#7f7f7f',
                  '#bcbc22',
                  '#1717cf']

        proc['metadata']['protocolName'] = proc_sid
        proc['metadata']['description'] = desc or ""
        proc['metadata']['author'] = self.configs.proj_owner
        proc['metadata']['created'] = round(time.time())
        proc['metadata']['lastModified'] = round(time.time())

        src_id_lut = {sid: str(i) for i, sid in enumerate(all_src_sid)}
        color_lut = {sid: colors[i % len(colors)] for i, sid in enumerate(all_src_sid)}
        ingredients = {src_id_lut[sid]: {
            'name': sid, 'description': '', 'serialize': False, 'liquidGroupId': src_id_lut[sid],
            'displayColor': color_lut[sid]
        } for sid in all_src_sid}
        proc['designerApplication']['data']['ingredients'] = ingredients

        proc['liquids'] = {src_id_lut[sid]: {
            'displayName': sid, 'description': '', 'displayColor': color_lut[sid]
        } for sid in all_src_sid}


        labware_location_update = {
            labware_item.uuid_path: str(slot) for slot, labware_item in designer.labware_map.items()
        }
        labware_location_update['fixedTrash'] = { "slotName": "12" }
        proc['designerApplication']['data']['savedStepForms']['__INITIAL_DECK_SETUP_STEP__'][
            'labwareLocationUpdate'] = labware_location_update

        # labware = {
        #     labware_item.uuid_path: {
        #         'slot': str(slot),
        #         'displayName': labware_item.display_name,
        #         'definitionId': labware_item.definition_id
        #     } for slot, labware_item in designer.labware_map.items()
        # }
        # proc['labware'] = labware

        pipette_ids = {
            MAIN_PIPETTE: 'be3a0e70-f850-4d16-9b37-d8c3f94e0177',
            ALT_PIPETTE: '8ac7f1e9-fc8a-41f6-a681-70d933edf799',
        }

        ingred_locations = {}
        for rack, v in designer.init_wells_snapshot.items():
            rack_id = designer.labware_map[rack].uuid_path
            ingred_locations[rack_id] = {well: {src_id_lut[sample.sid]: {'volume': sample.vol}}
                                         for well, sample in v.items() if sample.sid is not None}
            if len(ingred_locations[rack_id]) == 0:
                del ingred_locations[rack_id]

        proc['designerApplication']['data']['ingredLocations'] = ingred_locations

        commands = []
        step_pd = pd.DataFrame(
            columns=['idx', 'uid', 'step_num', 'type', 'slot', 'well', 'sid', 'vol', 'dst_tot_vol', 'rate', 'offset',
                     'delay', 'pipette', 'message'])
        
        if self.configs.use_alt_pipette:
            commands.append({
                    "key": "3501e89c-f4d6-4a66-993a-69a65720f1b1",
                    "commandType": "loadPipette",
                    "params": {
                        "pipetteName": "p300_single_gen2",
                        "mount": "left",
                        "pipetteId": pipette_ids[ALT_PIPETTE]
                    }
                }
            )
            
        if self.configs.use_main_pipette:
            commands.append({
                    "key": "72d1b575-c7aa-46e6-a093-9f113c2efe7f",
                    "commandType": "loadPipette",
                    "params": {
                        "pipetteName": "p1000_single_gen2",
                        "mount": "right",
                        "pipetteId": pipette_ids[MAIN_PIPETTE]
                    }
                }
            )

        for slot, labware_item in designer.labware_map.items():
            if labware_item.definition_id == 'fixedTrash': continue
            namespace, load_name, version = labware_item.definition_id.split('/')
            commands.append({
                "key": self._get_command_uuid(),
                "commandType": "loadLabware",
                "params": {
                    "displayName": labware_item.display_name,
                    "labwareId": labware_item.uuid_path,
                    "loadName": load_name,
                    "namespace": namespace,
                    "version": int(version),
                    "location": {"slotName": str(slot)}
                }
            })

        for rack, v in designer.init_wells_snapshot.items():
            rack_id = designer.labware_map[rack].uuid_path
            for well, sample in v.items():
                if sample.sid is None: continue
                commands.append({
                    "key": self._get_command_uuid(),
                    "commandType": "loadLiquid",
                    "params": {
                        "liquidId": src_id_lut[sample.sid],
                        "labwareId": rack_id,
                        "volumeByWell": {well: sample.vol}
                    }
                })

        for i, step in enumerate(designer.steps):
            if isinstance(step, AEDropTipStep):
                command_uuid = self._get_command_uuid()
                command = [{
                    'commandType': 'dropTip',
                    'key': command_uuid,
                    'params': {
                        'pipetteId': pipette_ids[step.pipette],
                        'labwareId': designer.labware_map[12].uuid_path,
                        'wellName': 'A1',
                    }
                }, ]
                step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'dropTip', 12, 'A1', '', na, na, na, na, na,
                                             step.pipette, '']
            elif isinstance(step, AEPickUpTipStep):
                command_uuid = self._get_command_uuid()
                command = [{
                    'commandType': 'pickUpTip',
                    'key': command_uuid,
                    'params': {
                        'pipetteId': pipette_ids[step.pipette],
                        'labwareId': designer.labware_map[step.src_slot].uuid_path,
                        'wellName': step.src_well,
                    }
                }, ]
                step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'pickUpTip', step.src_slot, step.src_well,
                                             '', na, na, na,
                                             na, na, step.pipette, '']
            elif isinstance(step, AEPauseStep):
                command_uuid = self._get_command_uuid()
                command = [{
                    'commandType': 'waitForResume',
                    'key': command_uuid,
                    'params': {
                        'message': step.message
                    }
                }, ]
                step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'waitForResume', 0, '', '', na, na, na, na, na,
                                             na, step.message]
            elif isinstance(step, AETransferStep):
                if step.src_sid in self.configs.skip_srcs:
                    continue

                d_offset = min(self.configs.dispense_offset_max,
                               step.dst_tot_vol * self.configs.dispense_offset_a + self.configs.dispense_offset_b)
                a_offset = self.configs.aspirate_offset

                if step.src_sid in self.configs.alt_aspirate_rate_srcs:
                    aspirate_rate = self.configs.alt_aspirate_rate
                else:
                    aspirate_rate = self.configs.aspirate_rate

                if step.src_sid in self.configs.alt_dispense_rate_srcs:
                    dispense_rate = self.configs.alt_dispense_rate
                else:
                    dispense_rate = self.configs.dispense_rate
                command = []

                if (self.configs.src_mixing_rounds > 0) and (step.src_sid not in self.configs.skip_src_mixing_srcs):
                    mixing_offset = self.configs.src_mixing_offset if self.configs.src_mixing_offset > 0 else a_offset
                    mixing_vol = self.configs.src_mixing_vol if self.configs.src_mixing_vol > 0 else step.pipette
                    mixing_rate = self.configs.src_mixing_rate if self.configs.src_mixing_rate > 0 else aspirate_rate
                    mixing_delay = self.configs.src_mixing_delay
                    for _ in range(self.configs.src_mixing_rounds):
                        command_uuid = self._get_command_uuid()
                        command.append({
                            'commandType': 'aspirate',
                            'key': command_uuid,
                            'params': {
                                'pipetteId': pipette_ids[step.pipette],
                                'volume': mixing_vol,
                                'labwareId': designer.labware_map[step.src_slot].uuid_path,
                                'wellName': step.src_well,
                                'wellLocation': {"origin": "bottom", "offset": {"z": mixing_offset}},
                                'flowRate': mixing_rate,
                            }
                        })
                        step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'aspirate', step.src_slot,
                                                     step.src_well,
                                                     step.src_sid,
                                                     mixing_vol,
                                                     na,
                                                     mixing_rate, mixing_offset, na, step.pipette, '']
                        if mixing_delay > 0:
                            command_uuid = self._get_command_uuid()
                            command.append({
                                'commandType': 'waitForDuration',
                                'key': command_uuid,
                                'params': {
                                    'seconds': mixing_delay,
                                    'message': 'mixing delay'
                                }
                            })
                            step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na,
                                                         na, na,
                                                         mixing_delay, na, '']
                        command_uuid = self._get_command_uuid()
                        command.append({
                            'commandType': 'dispense',
                            'key': command_uuid,
                            'params': {
                                'pipetteId': pipette_ids[step.pipette],
                                'volume': mixing_vol,
                                'labwareId': designer.labware_map[step.src_slot].uuid_path,
                                'wellName': step.src_well,
                                'wellLocation': {"origin": "bottom", "offset": {"z": mixing_offset}},
                                'flowRate': mixing_rate,
                            }
                        })
                        step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'dispense', step.src_slot,
                                                     step.src_well,
                                                     step.src_sid,
                                                     mixing_vol,
                                                     na,
                                                     mixing_rate, mixing_offset, na, step.pipette, '']
                        if mixing_delay > 0:
                            command_uuid = self._get_command_uuid()
                            command.append({
                                'commandType': 'waitForDuration',
                                'key': command_uuid,
                                'params': {
                                    'seconds': mixing_delay,
                                    'message': 'mixing delay'
                                }
                            })
                            step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na,
                                                         na, na,
                                                         mixing_delay, na, '']

                command_uuid = self._get_command_uuid()
                command.append({
                    'commandType': 'aspirate',
                    'key': command_uuid,
                    'params': {
                        'pipetteId': pipette_ids[step.pipette],
                        'volume': step.vol,
                        'labwareId': designer.labware_map[step.src_slot].uuid_path,
                        'wellName': step.src_well,
                        'wellLocation': {"origin": "bottom", "offset": {"z": a_offset}},
                        'flowRate': aspirate_rate,
                    }
                })
                step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'aspirate', step.src_slot, step.src_well,
                                             step.src_sid,
                                             step.vol,
                                             na,
                                             aspirate_rate, a_offset, na, step.pipette, '']

                if self._need_delay('aspirate', step.src_sid):
                    command_uuid = self._get_command_uuid()
                    command.append({
                        'commandType': 'waitForDuration',
                        'key': command_uuid,
                        'params': {
                            'seconds': self.configs.aspirate_delay,
                            'message': 'aspirate delay'
                        }
                    })
                    step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na, na, na,
                                                 self.configs.aspirate_delay, na, '']

                if step.pipette == MAIN_PIPETTE:
                    air_gap_vol = self.configs.air_gap_vol
                elif step.pipette == ALT_PIPETTE:
                    air_gap_vol = self.configs.alt_air_gap_vol
                else:
                    raise ValueError('Unknown pipette.')
                if air_gap_vol > 0:
                    command_uuid = self._get_command_uuid()
                    command.append(
                        {
                            'commandType': 'aspirate',
                            'key': command_uuid,
                            'params': {
                                'pipetteId': pipette_ids[step.pipette],
                                'volume': air_gap_vol,
                                'labwareId': designer.labware_map[step.src_slot].uuid_path,
                                'wellName': step.src_well,
                                'wellLocation': {"origin": "bottom", "offset": {"z": self.configs.air_gap_offset}},
                                'flowRate': aspirate_rate,
                            },
                            "meta": {"isAirGap": True}
                        })
                    step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'aspirate', step.src_slot,
                                                 step.src_well, 'AIR-0000',
                                                 air_gap_vol,
                                                 na,
                                                 aspirate_rate, self.configs.air_gap_offset, na, step.pipette, '']

                    if self._need_delay('airgap', step.src_sid):
                        command_uuid = self._get_command_uuid()
                        command.append({
                            'commandType': 'waitForDuration',
                            'key': command_uuid,
                            'params': {
                                'seconds': self.configs.airgap_delay,
                                'message': 'airgap delay'
                            }
                        })
                        step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na, na, na,
                                                     self.configs.airgap_delay, na, '']

                    command_uuid = self._get_command_uuid()
                    command.append({
                        'commandType': 'dispense',
                        'key': command_uuid,
                        'params': {
                            'pipetteId': pipette_ids[step.pipette],
                            'labwareId': designer.labware_map[step.dst_slot].uuid_path,
                            "volume": air_gap_vol,
                            'wellName': step.dst_well,
                            'wellLocation': {"origin": "bottom", "offset": {"z": d_offset}},
                            'flowRate': dispense_rate,
                        },
                        "meta": {"isAirGap": True}
                    })
                    step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'dispense', step.dst_slot,
                                                 step.dst_well,
                                                 step.dst_sid,
                                                 step.vol + air_gap_vol,
                                                 step.dst_tot_vol,
                                                 dispense_rate, d_offset, na, step.pipette, '']

                command_uuid = self._get_command_uuid()
                command.append({
                    'commandType': 'dispense',
                    'key': command_uuid,
                    'params': {
                        'pipetteId': pipette_ids[step.pipette],
                        'labwareId': designer.labware_map[step.dst_slot].uuid_path,
                        "volume": step.vol,
                        'wellName': step.dst_well,
                        'wellLocation': {"origin": "bottom", "offset": {"z": d_offset}},
                        'flowRate': dispense_rate,
                    },
                })
                if self._need_delay('dispense', step.src_sid):
                    command_uuid = self._get_command_uuid()
                    command.append({
                        'commandType': 'waitForDuration',
                        'key': command_uuid,
                        'params': {
                            'seconds': self.configs.dispense_delay,
                            'message': 'dispense delay'
                        }
                    })
                    step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na, na, na,
                                                 self.configs.dispense_delay, na, '']

                command_uuid = self._get_command_uuid()
                command.append({
                    'commandType': 'blowout',
                    'key': command_uuid,
                    'params': {
                        'pipetteId': pipette_ids[step.pipette],
                        'labwareId': designer.labware_map[step.dst_slot].uuid_path,
                        'wellName': step.dst_well,
                        'wellLocation': {"origin": "bottom", "offset": {"z": d_offset}},
                        'flowRate': dispense_rate,
                    },
                })
                step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'blowout', step.dst_slot, step.dst_well,
                                             step.dst_sid,
                                             na,
                                             na,
                                             dispense_rate, d_offset, na, step.pipette, '']

                if self._need_delay('blowout', step.src_sid):
                    command_uuid = self._get_command_uuid()
                    command.append({
                        'commandType': 'waitForDuration',
                        'key': command_uuid,
                        'params': {
                            'seconds': self.configs.blowout_delay,
                            'message': 'blowout delay'
                        }
                    })

                    step_pd.loc[len(step_pd)] = [len(step_pd), command_uuid, i, 'delay', na, na, na, na, na, na, na,
                                                 self.configs.blowout_delay, na, '']

            else:
                raise ValueError('Unknown step.')
            commands.extend(command)

        proc['commands'] = commands

        with open(self.get_asset_path(proc_sid, 'ot2_proc'), 'w') as f:
            json.dump(proc, f, indent=2, ensure_ascii=True)

        # Export OT2 steps table (for debug only)
        csv_path = self.get_asset_path(proc_sid, 'debug_ot2_steps')
        step_pd.to_csv(csv_path, index=False)

        # Validate steps
        try:
            if self.configs.use_main_pipette:
                assert step_pd[(step_pd['type'] == 'aspirate') & (step_pd['pipette'] == MAIN_PIPETTE)][
                           'vol'].max() <= self.configs.pipette_max_vol
                assert step_pd[(step_pd['type'] == 'aspirate') & (step_pd['pipette'] == MAIN_PIPETTE)][
                           'vol'].min() >= self.configs.pipette_min_vol
            if self.configs.use_alt_pipette:
                assert step_pd[(step_pd['type'] == 'aspirate') & (step_pd['pipette'] == ALT_PIPETTE)][
                           'vol'].max() <= self.configs.alt_pipette_max_vol
                assert step_pd[(step_pd['type'] == 'aspirate') & (step_pd['pipette'] == ALT_PIPETTE)][
                           'vol'].min() >= self.configs.alt_pipette_min_vol

            assert step_pd[step_pd['type'] == 'aspirate']['offset'].min() >= 0.5
            assert step_pd[step_pd['type'] == 'aspirate']['offset'].max() <= 130
            assert step_pd[step_pd['type'] == 'dispense']['offset'].min() >= 0.5
            assert step_pd[step_pd['type'] == 'dispense']['offset'].max() <= 130
        except AssertionError as e:
            print(e)

    def _gen_ot2_sources(self, proc_sid: str, designer: OT2Designer):
        src_pd = pd.DataFrame(columns=['slot', 'well', 'sID', 'con', 'vol', 'part_i', 'n_parts'])

        for slot, v in designer.init_wells_snapshot.items():
            for well, source in v.items():
                if source.sid is not None:
                    if source.sid in self.configs.sources:
                        con_str = self.configs.sources[source.sid]['concentration']
                    else:
                        con_str = ''
                    part_i = designer.seperated_src_reverse_map[source.sid] if designer.separate_parts else 1
                    src_pd.loc[len(src_pd)] = [slot, well, source.sid, con_str, source.vol, part_i,
                                               designer.n_parts]

        # src_pd.loc[len(src_pd)] = [1, self.configs.first_tip, '_FIRST_TIP_USED_', 0, 0, 0, 0]
        # src_pd.loc[len(src_pd)] = [*designer.tip_wells[0], '_FIRST_TIP_REMAINING_', 0, 0, 0, 0]

        # Export OT2 source layout table
        src_pd.to_csv(self.get_asset_path(proc_sid, 'debug_sources'), index=False)

    def _gen_labels(self, proc_sid):
        well_df = pd.read_csv(self.get_asset_path(proc_sid, 'debug_ot2_wells'),
                              dtype={'additional_flags': str}, keep_default_na=False)
        source_df = pd.read_csv(self.get_asset_path(proc_sid, 'debug_sources'))

        label_types = str_to_list(self.configs.label_types)
        owner = self.configs.proj_owner

        job_list = []
        for label_type in label_types:
            if 'ot2_src' == label_type:
                for _, row in source_df.iterrows():
                    if row.vol > 0:
                        job_list.append((label_type, 'src_tube',
                                         dict(sid=row.sID,
                                              ot2_proc_sid=proc_sid,
                                              src_vol=row.vol,
                                              src_con=row.con,
                                              slot=row.slot,
                                              well=row.well,
                                              owner=owner,
                                              part_i=row.part_i,
                                              n_parts=row.n_parts,
                                              )))

            elif 'ot2_dst' == label_type:
                for _, row in well_df.iterrows():
                    job_list.append((label_type, 'dst_tube',
                                     dict(sid=row.sample_sid,
                                          ot2_proc_sid=proc_sid,
                                          slot=row.ot2_slot,
                                          well=row.ot2_well,
                                          owner=owner,
                                          part_i=row.part_i,
                                          n_parts=row.n_parts,
                                          )))

            elif 'ot2_dst_sep' == label_type:
                for _, row in well_df.iterrows():
                    job_list.append((label_type, 'dst_tube_sep',
                                     dict(sid=row.sample_sid,
                                          ot2_proc_sid=proc_sid,
                                          slot=row.ot2_slot,
                                          well=row.ot2_well,
                                          owner=owner)))

            elif 'oven_boat' == label_type:
                for _, row in well_df.iterrows():
                    if row.part_i > 1: continue
                    job_list.append((label_type, 'oven_boat',
                                     dict(sid=row.sample_sid,
                                          owner=owner,
                                          slot=row.ot2_slot)))

            elif 'dish_comp' == label_type:
                for _, row in well_df.iterrows():
                    if row.part_i > 1: continue
                    d = dict(sid=row.sample_sid, owner=owner, slot=row.ot2_slot,
                             additional_flags=row.additional_flags, mass=row.total_mass)
                    non_zero_i = 0
                    #                     print(row.sample_sid, self.configs.n_sources_per_sample)
                    for i in range(1, self.configs.n_sources_per_sample + 1):
                        if getattr(row, f'src{i}_vol', 0) > 0:
                            non_zero_i += 1
                            d[f'src{non_zero_i}_sid'] = getattr(row, f'src{i}_sid', None)
                            d[f'src{non_zero_i}_frc'] = getattr(row, f'src{i}_frc', None)
                            d[f'src{non_zero_i}_vol'] = getattr(row, f'src{i}_vol', None)
                    if non_zero_i > 10:
                        auto_log(f'Max number of sources that a label can display is 10, but sample contains '
                                 f'{non_zero_i} sources.')
                    job_list.append((label_type, 'dish_comp', d))

            elif 'dish_comp_barcode' == label_type:
                for _, row in well_df.iterrows():
                    if row.part_i == 1:
                        d = dict(sid=row.sample_sid, owner=owner, slot=row.ot2_slot,
                                 additional_flags=row.additional_flags, ot2_proc_sid=proc_sid)
                        non_zero_i = 0
                        for i in range(1, self.configs.n_sources_per_sample + 1):
                            if getattr(row, f'src{i}_vol', 0) > 0:
                                non_zero_i += 1
                                d[f'src{non_zero_i}_sid'] = getattr(row, f'src{i}_sid', None)
                                d[f'src{non_zero_i}_frc'] = getattr(row, f'src{i}_frc', None)
                                d[f'src{non_zero_i}_vol'] = getattr(row, f'src{i}_vol', None)
                        if non_zero_i > 10:
                            warnings.warn(f'Max number of sources that a label can display is 10, but sample contains '
                                          f'{non_zero_i} sources.', UserWarning)
                        job_list.append((label_type, 'dish_comp_barcode', d))
            elif 'dish_comp_half' == label_type:
                for _, row in well_df.iterrows():
                    if row.part_i == 1:
                        d = dict(sid=row.sample_sid, owner=owner, slot=row.ot2_slot, well=row.ot2_well,
                                 additional_flags=row.additional_flags, ot2_proc_sid=proc_sid)
                        non_zero_i = 0
                        for i in range(1, self.configs.n_sources_per_sample + 1):
                            if getattr(row, f'src{i}_vol', 0) > 0:
                                non_zero_i += 1
                                d[f'src{non_zero_i}_sid'] = getattr(row, f'src{i}_sid', None)
                                d[f'src{non_zero_i}_frc'] = getattr(row, f'src{i}_frc', None)
                                d[f'src{non_zero_i}_vol'] = getattr(row, f'src{i}_vol', None)
                        if non_zero_i > 10:
                            warnings.warn(f'Max number of sources that a label can display is 10, but sample contains '
                                          f'{non_zero_i} sources.', UserWarning)
                        job_list.append((label_type, 'dish_comp_half', d))

            elif 'barcode' == label_type:
                for _, row in well_df.iterrows():
                    if row.part_i > 1: continue
                    job_list.append(
                        (label_type, 'barcode',
                         dict(sid=row.sample_sid, owner=owner, slot=row.ot2_slot,
                              ot2_proc_sid=proc_sid)))
            elif 'PAGE_BREAK' == label_type:
                job_list.append(('PAGE_BREAK', None, None))
            elif 'COLUMN_BREAK' == label_type:
                job_list.append(('COLUMN_BREAK', None, None))
            elif 'SPACE_BREAK' == label_type:
                job_list.append(('SPACE_BREAK', None, None))
            else:
                raise ValueError(f'Unknown label type: {label_type}')

        n_pages, page_pdf_list, axes = 0, [], None

        # if self.configs.label_page_break_between_groups
        if self.configs.label_grouping == 'by_type':
            group_key = lambda j: j[0]
            sort_key = lambda j: label_types.index(j[0])

        elif self.configs.label_grouping == 'by_slot':
            src_slots = self.configs.src_slots.split(',')
            group_key = lambda j: j[2]['slot'] if str(j[2]['slot']) not in src_slots else 99
            sort_key = group_key
        elif self.configs.label_grouping == 'none':
            group_key = lambda j: 1
            sort_key = group_key
        else:
            raise ValueError(f'Unknown label_grouping: {self.configs.label_grouping}')

        i_total = 0
        for group_name, group_iter in itertools.groupby(sorted(job_list, key=sort_key), group_key):
            grouped_jobs = list(group_iter)
            for i_group, (label_type, label_style, kwargs) in enumerate(grouped_jobs):
                if axes is None:
                    n_pages += 1
                    lm, fig, axes = new_label_page(tmpl=self.configs.label_template)

                    text = f'{proc_sid}.ot2_proc.json\n' + \
                           f'batch = {self.configs.batch_sid}\n' + \
                           f'{self.configs.batch_desc}\n' + \
                           f'page {n_pages}'
                    lm.draw_label(axes.pop(0), 'text', text=text)
                if label_type == 'PAGE_BREAK':
                    axes = []
                elif label_type == 'COLUMN_BREAK':
                    remaining_full_columns = len(axes) // lm.n_labels_per_column
                    if remaining_full_columns > 0:
                        axes = axes[-remaining_full_columns * lm.n_labels_per_column:]
                    else:
                        axes = []
                elif label_type == 'SPACE_BREAK':
                    axes.pop(0)
                else:
                    lm.draw_label(axes.pop(0), label_style, **kwargs)

                if i_group == len(grouped_jobs) - 1:
                    if self.configs.label_break_between_groups == 'page':
                        axes = []
                    elif self.configs.label_break_between_groups == 'column':
                        remaining_full_columns = len(axes) // lm.n_labels_per_column
                        if remaining_full_columns > 0:
                            axes = axes[-remaining_full_columns * lm.n_labels_per_column:]
                        else:
                            axes = []
                    else:
                        pass

                if len(axes) <= 0 or i_total == len(job_list) - 1:
                    pdf_path = self.get_asset_path(proc_sid, 'labels') + f'.p{n_pages}.pdf'
                    plt.savefig(pdf_path)
                    page_pdf_list.append(pdf_path)
                    plt.close('all')
                    axes = None

                i_total += 1

        pdf_path = self.get_asset_path(proc_sid, 'labels')
        merger = PdfMerger()
        for pdf in page_pdf_list:
            merger.append(pdf)
        merger.write(pdf_path)
        merger.close()
        for pdf in page_pdf_list:
            os.remove(pdf)

    def get_asset_path(self, proc_sid, item_name):
        assets = {
            'ot2_proc': 'ot2_proc.json' if self.configs.proc_filename is None else f'{self.configs.proc_filename}.json',
            'labels': 'labels.pdf' if self.configs.label_filename is None else f'{self.configs.label_filename}.pdf',
            'debug_ot2_steps': 'debug.ot2_steps.csv',
            'debug_ot2_wells': 'debug.ot2_wells.csv',
            'debug_sources': 'debug.sources.csv',
            'debug_samples': 'debug.samples.csv',
            'debug_configs': 'debug.configs.yml'
        }
        if self.out_dir:
            out_dir = Path(self.out_dir) / proc_sid
        else:
            out_dir = OT2_PROC_DIR / proc_sid
        os.makedirs(out_dir, exist_ok=True)
        return str(out_dir / f'{proc_sid}.{assets[item_name]}')
