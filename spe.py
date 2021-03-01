import argparse
import copy
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
# ORTEC Software File Structure Manual for DOS and Windows® Systems
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
            sys.exit(f"Could not find file: {filepath}")
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
        self.run_time = timedelta(seconds=int(self.meas_tim[0]))
        self.end_time = self.date + self.run_time
        self.data = self.blocks["$DATA:"]
        self.roi = "Does not parse ROI at the moment."
        self.presets = "\n".join(self.blocks["$PRESETS:"])
        self.ener_fit= tuple([float(i) for i in self.blocks["$ENER_FIT:"][0].split(" ")])
        self.mca_cal = f'Number of energy calibration factors = ' \
                       f'{self.blocks["$MCA_CAL:"][0]}\n' \
                       f'{self.blocks["$MCA_CAL:"][1]}'
        self.shape_cal = f'Number of shape (FWHM) calibration factors = ' \
                       f'{self.blocks["$SHAPE_CAL:"][0]}\n' \
                       f'{self.blocks["$SHAPE_CAL:"][1]}'
        bins = np.array(range(len(self.data)))
        self.calib = tuple(float(i) for i in self.blocks["$MCA_CAL:"][1].split(" ")[:-1])
        self.ebins = self.calib[0] + self.calib[1] * bins + self.calib[2] * bins * bins

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
        if keyword == "$DATA:":
            block_data = block_data[1:].astype(int)
            return block_data
        else:
            return block_data

    def scale(self, factor: float) -> "SPE":
        scaled_data = self.data * factor
        new_spe = copy.deepcopy(self)
        new_spe.data = scaled_data
        #plt.scatter(self.ebins, self.data, s=0.2)
        #plt.scatter(new_spe.ebins, new_spe.data, s=0.2)
        #plt.show()
        return new_spe

    def subtract(self, background: "SPE", bin_range: int=40) -> "SPE":
        if self.calib != background.calib:
            raise ValueError("Background calibration does not match.")
        best_quality = 1000000000000
        best_result = None
        best_bins = None
        best_offset = None
        for bin_offset in range(-bin_range, bin_range+1):
            if bin_offset < 0:
                reduced = self.data[-bin_offset:] - background.data[:bin_offset]
                bins = self.ebins[-bin_offset:]
            elif bin_offset > 0:
                reduced = self.data[:-bin_offset] - background.data[bin_offset:]
                bins = self.ebins[:-bin_offset]
            else:
                reduced = self.data - background.data
                bins = self.ebins
            assert len(reduced) == len(self.data) - np.abs(bin_offset)
            assert len(bins) == len(self.ebins) - np.abs(bin_offset)
            #quality = np.count_nonzero(reduced < 0)
            quality = sum(reduced[reduced<0]) * -1
            #print(f"Offset: {bin_offset}. Sum of bins < 0: {quality}")
            if quality < best_quality: 
                best_offset = bin_offset
                best_quality = quality
                best_result = reduced
                best_bins = bins
        print(f"Best offset is {best_offset}")
        new_spe = copy.deepcopy(self)
        new_spe.data = best_result
        new_spe.ebins = best_bins
        return new_spe

    def subtract_centered(
        self, background: "SPE",
        centre: float,
        bin_range: int=40,
        check_range: int=50,
        check: bool=True
        ) -> "SPE":
        # Match background using a peak at wavelength "centre"
        s_selection = np.where((self.ebins > centre - check_range) &
                               (self.ebins < centre + check_range))[0]
        s_bin_no = s_selection[np.argmax(self.data[s_selection])]
        b_selection = np.where((background.ebins > centre - check_range) &
                               (background.ebins < centre + check_range))[0]
        b_bin_no = b_selection[np.argmax(background.data[b_selection])]
        bin_offset = s_bin_no - b_bin_no
        if check:
            plt.plot(self.ebins, self.data, linewidth=0.5)
            plt.axvline(self.ebins[s_bin_no], c='r', linewidth=0.5)
            plt.axvline(int(centre-check_range), c='orange', linewidth=0.5, linestyle="--")
            plt.axvline(int(centre+check_range), c='orange', linewidth=0.5, linestyle="--")
            plt.show()
            plt.plot(background.ebins, background.data, linewidth=0.5)
            plt.axvline(background.ebins[b_bin_no], c='r', linewidth=0.5)
            plt.axvline(int(centre-check_range), c='orange', linewidth=0.5, linestyle="--")
            plt.axvline(int(centre+check_range), c='orange', linewidth=0.5, linestyle="--")
            plt.show()
            answer = input("Was the red line at the correct peak? (y/n):\n")
            if answer != 'y':
                raise AssertionError("Incorrect peak identified")
        print(f"Offset at wavelength {centre} (bin {s_bin_no}) is {bin_offset}")
        if bin_offset < 0:
            reduced = self.data[-bin_offset:] - background.data[:bin_offset]
            bins = self.ebins[-bin_offset:]
        elif bin_offset > 0:
            reduced = self.data[:-bin_offset] - background.data[bin_offset:]
            bins = self.ebins[:-bin_offset]
        else:
            reduced = self.data - background.data
            bins = self.ebins
        new_spe = copy.deepcopy(self)
        new_spe.data = reduced
        new_spe.ebins = bins
        return new_spe

    def smooth(self, window: int=10) -> "SPE":
        bin_cumsum = np.cumsum(self.data)
        moving_average = (bin_cumsum[window:] - bin_cumsum[:-window]) / window
        #plt.plot(self.ebins, self.data, linewidth=0.5)
        #plt.plot(self.ebins[int(window/2):int(-window/2)], moving_average, linewidth=0.5)
        #plt.show()
        new_spe = copy.deepcopy(self)
        new_spe.data = moving_average
        new_spe.ebins = self.ebins[int(window/2):int(-window/2)]
        return new_spe

    def save(self) -> None:
        data_to_save = np.asarray([self.ebins, self.data]).T
        fn, _ = os.path.splitext(os.path.abspath(self.filepath))
        np.savetxt(fn+"_subtracted"+".csv", data_to_save, delimiter=",", header="energy (keV), counts")

# Since subtract returns a new SPE, this should be instance method. Can pass
# title with info on sample and duration using instance attributes.
def plot_spectrum(spe, peaks=False, threshold=0, width=0, height=0,
                  distance=0, prominence=0, save=False) -> None:
    alpha = 1
    if peaks:
        peaks, _ = find_peaks(spe.data, threshold=threshold, height=height,
                              width=width, distance=distance,
                              prominence=prominence)
        plt.plot(spe.ebins[peaks], spe.data[peaks], "x", c="orange")
        alpha=0.5
    plt.plot(spe.ebins, spe.data, linewidth=0.5)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Counts")
    plt.ylim(0, max(spe.data))
    plt.title(f"{spe.filepath}")
    if save:
        spe.save()
        fn, _ = os.path.splitext(os.path.abspath(spe.filepath))
        figure = plt.gcf()
        figure.set_size_inches(4, 3)
        plt.savefig(fn + ".png", bbox_inches="tight", dpi=200)
    plt.show()

def find_background(spe: SPE) -> SPE:
    # Attempt to find appropriate background for given SPE.
    bg_dir = os.path.abspath(f"{spe.filepath}/../../../Backgrounds")
    print(f"Found background directory: {bg_dir}")
    for fn in os.listdir(bg_dir):
        print("Found possible background file.")
        bg_spe = SPE(os.path.abspath(bg_dir + "/" + fn))
        if bg_spe.calib == spe.calib:
            print(f"Found matching background: {bg_spe.filepath}")
            return bg_spe
        else:
            print("This background file doesn't match.")
    print("Could not find matching background.")
    return None

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Filepath to .spe file", type=str)
    parser.add_argument("-s", "--subtract", help="Subtract specified .spe file",
                        type=str)
    parser.add_argument("-p", "--plot", help="Plot spectrum", action="store_true")
    parser.add_argument("--save", help="Save plot and data. Requires --plot", action="store_true")
    parser.add_argument("--smooth", help="Smooth the data", action="store_true")
    parser.add_argument("--centre",
                        help="Centre background subtraction on peak at wavelength [centre]",
                        type=float)
    args = parser.parse_args(argv)
    if not os.path.exists(args.file):
        parser.error(f"File {args.file} does not exist.")
    return args

if __name__ == "__main__":
    options = parse_args()
    signal = SPE(options.file)
    spe_out = None
    if options.subtract is not None:
        if options.subtract == "":
            print("No background file provided, searching.")
            background = find_background(signal)
        else:
            bg_file = options.subtract
            background = SPE(bg_file)
        if options.smooth:
            signal = signal.smooth()
            background = background.smooth()
        scaled_background = background.scale(signal.run_time / background.run_time)
        if options.centre:
            signal.subtract_centered(scaled_background, options.centre)
        else:
            spe_out = signal.subtract(scaled_background)
    else:
        spe_out = signal
    if options.plot:
        plot_spectrum(spe_out, save=options.save) #peaks=True, prominence=50, distance=1000)#height=15, threshold=1, distance=80, width=8)
