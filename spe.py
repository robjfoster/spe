from datetime import datetime, timedelta
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

if sys.version_info[0] < 3 or sys.version_info[1] < 6:
    sys.exit("Requires python > 3.6")

# The keywords used to delimit the sections of the output file.
# Descriptions taken from: 
# ORTEC Software File Structure Manualfor DOS and Windows® Systems
keywords = {
    "$SPEC_ID:": "One line of text describing the data",
    "$SPEC_REM:": "Any number of lines containing remarks about the data",
    "$DATE_MEA:": "Measurement date in the form mm/dd/yyyy hh:mm:ss",
    "$MEAS_TIM:": "Live time and realtime of the spectrum in integer seconds, "
                  "separated by spaces",
    "$DATA:": "The first line contains the channel number of the first channel "
              "and the number of channels separated by spaces. The remaining "
              "lines contain one channel each of data.",
    "$ROI:": "This group contains the regions of interest marked in the "
             "spectrum. The firstline the number of regions, the following "
             "lines contain the start and stop channels for each region.",
    "$PRESETS:": "No description",
    "$ENER_FIT:": "This contains the energy calibration factors (a + b * chn) "
                  "as two real numbers, separated by spaces.",
    "$MCA_CAL:": "This contains the number of energy calibration factors on "
                 "the first line, then the factors on the second line as "
                 "two numbers, separated by spaces.",
    "$SHAPE_CAL:": "This contains the number of FWHM calibration factors on "
                   "the first line,then the factors on the second line as "
                   "two numbers, separated by spaces.",
    
}

class SPE():
    
    def __init__(self, filepath: str) -> None:
        if not os.path.isfile(filepath):
            sys.exit(f"Could not find {filepath}")
        self.filepath = filepath
        self.blocks = {}
        self.data = np.array([])
        self.rawdata = np.loadtxt(filepath, dtype=str, delimiter="\n", 
                                  comments=None)
        print(f"\nReading in file {filepath}", end=" -> ")
        try:
            self.parse()
        except:
            breakpoint()
            raise ValueError("Failed to parse file")
        print("success\n")
        print(self.preamble())
        
    def parse(self) -> None:
        row_count = len(self.rawdata)
        delim_rows = {}
        for row, value in enumerate(self.rawdata):
            if value in keywords:
                delim_rows[value] = row    
        next_delim_rows = {}
        for kw, row in self.gen_block_end(delim_rows, row_count):
            next_delim_rows[kw] = row
        data_blocks = {}
        for kw in delim_rows:
            start = delim_rows[kw] + 1
            end = next_delim_rows[kw]
            data_block = (start, end)
            data_blocks[kw] = data_block
        for kw in keywords:
            self.blocks[kw] = self.get_block(kw, data_blocks)
        self.id = str(self.blocks["$SPEC_ID:"][0])
        self.rem = "\n".join(self.blocks["$SPEC_REM:"])
        self.date = datetime.strptime(self.blocks["$DATE_MEA:"][0], 
                                      '%m/%d/%Y %H:%M:%S')
        self.meas_tim = "".join(self.blocks["$MEAS_TIM:"]).split(" ")
        self.end_time = self.date + timedelta(seconds=int(self.meas_tim[0]))
        self.data = self.blocks["$DATA:"]
        self.counts = self.data
        self.roi = self.blocks["$ROI:"]
        self.presets = "\n".join(self.blocks["$PRESETS:"])
        self.ener_fit= tuple([float(i) for i in self.blocks["$ENER_FIT:"][0].split(" ")])
        self.mca_cal = f'Number of energy calibration factors = ' \
                       f'{self.blocks["$MCA_CAL:"][0]}\n' \
                       f'{self.blocks["$MCA_CAL:"][1]}'
        self.shape_cal = f'Number of shape (FWHM) calibration factors = ' \
                       f'{self.blocks["$SHAPE_CAL:"][0]}\n' \
                       f'{self.blocks["$SHAPE_CAL:"][1]}'
        bins = np.array(range(len(self.data)))
        self.ebins = self.ener_fit[0] + self.ener_fit[1] * bins
        
    def preamble(self) -> str:
        text = ""
        text += "File details:\n"
        text += f"\n---Sample description---\n{self.id}\n"
        text += f"\n---Remarks---\n{self.rem}\n"
        text += f"\n---Measurement---\n"
        text += f"Detector start time: {self.date}\n"
        text += f"Detector end time:   {self.end_time}\n"
        text += f"\nLive time (s): {self.meas_tim[0]}\n"
        text += f"Real time (s): {self.meas_tim[1]}\n"
        text += f"\n---Presets---\n{self.presets}\n"
        text += f"\n---Energy calibration---\n{self.mca_cal}\n"
        text += f"\n---Shape calibration---\n{self.shape_cal}\n"
        return text
            
    @staticmethod
    def gen_block_end(kw_row_dict: dict, rows: int):
    # Lookahead generator to find the final row of each data block
        it = iter(kw_row_dict)
        last = next(it)
        for val in it:
            yield last, kw_row_dict[val]
            last = val
        yield last, rows
        
    def get_block(self, keyword: str, data_blocks: dict) -> np.array:
        start, end = data_blocks[keyword]
        block_data = self.rawdata[start:end]
        if keyword == "$DATA:" or keyword == "$ROI:":
            block_data = block_data[1:].astype(int)
            return block_data
        else:
            return block_data
        
    def minus(self, background: np.array) -> np.array:
        return self.data - background.data
        
def plot_spectrum(energy, counts, peaks=False, threshold=0, width=0, height=0):
    alpha = 1
    if peaks:
        peaks, _ = find_peaks(counts, threshold=threshold, height=height, width=width)
        plt.plot(energy[peaks], counts[peaks], "x", c="orange")
        alpha=0.5
    plt.scatter(energy, counts, s=0.2, alpha=alpha)
    plt.show()
    

fn = "data/rocksalt_activation/rocksalt_sample2_run2.Spe"
spe = SPE(fn)
fn2 = "data/rocksalt_activation/rocksalt_sample2_2blocks.Spe"
spe2 = SPE(fn2)
