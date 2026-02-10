import os
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import copy

# Reproducibility control
seed_value = 42
np.random.seed(seed_value)

def fit_polynomial_ransac(data, degree, threshold):
    """Fit a polynomial to the data using RANSAC for outlier detection
    
    Args:
        - data: numpy array of shape (n_samples, 2), where the first column is x and the second is y.
        - degree: maximum degree of the polynomial to fit.
        - threshold: threshold for considering a point as an inlier.
    
    Returns:
        - coeffs: coefficients of the fitted polynomial.
    """
    
    # Separate the input data into x and y
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)
    
    # Create a RANSAC regressor
    model = RANSACRegressor(LinearRegression(), residual_threshold=threshold, random_state=seed_value)

    # Fit the model
    model.fit(X_poly, y)
    
    # Get the inlier mask
    inlier_mask = model.inlier_mask_
    
    # Fit the model again using only the inliers
    X_inliers = X_poly[inlier_mask]
    y_inliers = y[inlier_mask]
    model.fit(X_inliers, y_inliers)
    
    # Get the polynomial coefficients
    coeffs = model.estimator_.coef_
    intercept = model.estimator_.intercept_
    
    # The coefficients are returned in the form of a polynomial
    return np.concatenate(([intercept], coeffs[1:]))

class LaneDetector:
    """Lane points detector using sliding windows search on intensity histogram
    """ 
    def __init__(self, figure_dir="./", output_dir="./", distance_threshold=0.05, num_iterations=100, intensity_bin_res=0.2, horizontal_bin_res=0.8, lane_width=2, lane_bound_threshold=0.5, degree=3, num_lanes=2) -> None:
        """LaneDetector initializer

        Args:
            - figure_dir (str, optional): directory to save analysis figures (histograms & detected lanes). Default to "./".
            - output_dir (str, optional): directory to save coefficient outputs. Default to "./".
            - distance_threshold (float, optional): maximum distance from a point to the plane to be considered an inlier. Defaults to 0.05.
            - num_iterations (int, optional): the number of iterations that were needed for the sample consensus. Defaults to 100.
            - intensity_bin_res (float, optional): hyperparameter to setup intensity value bin. Defaults to 0.2.
            - horizontal_bin_res (float, optional): hyperparameter to setup y value bin. Defaults to 0.8.
            - lane_width (float, optional): hyperparameter to setup the baseline "width" between 2 lanes. Defaults to 2.
            - lane_bound_threshold (float, optional): hyperparameter to check the lane width is within the bound. Defaults to 0.5.
            - degree (int, optional): polynomial degree. Defaults to 3 (according to the assignment's requirement).
            - num_lanes (int, optional): number of lanes. Defaults to 2 (according to the assignment requirements).
        """        
        
        self.degree = degree  

        self.num_lanes = num_lanes
        
        self.distance_threshold = distance_threshold
        
        self.num_iterations = num_iterations
        
        self.intensity_bin_res = intensity_bin_res
        
        self.horizontal_bin_res = horizontal_bin_res
        
        self.lane_width = lane_width
        
        self.lane_bound_threshold = lane_bound_threshold
        
        # Make peak_figure_dir if it doesn't exist yet
        self.peak_figure_dir = os.path.join(figure_dir, "peaks")
        os.makedirs(self.peak_figure_dir, exist_ok=True)
        
        # Make lane_figure_dir if it doesn't exist yet
        self.lane_figure_dir = os.path.join(figure_dir, "lanes")
        os.makedirs(self.lane_figure_dir, exist_ok=True)
       
        # Make output_dir if it doesn't exist yet
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
                
    def read_pointcloud_data(self, input_path):
        """Read the pointcloud data from the input file

        Args:
            - data: input pointcloud data as shape (N, 5) (x, y, z, intensity, lidar_beam)
        """        
        
        self.points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)

    def extract_ground_points(self):
        """Find ground plane + ground points from the pointcloud points
        
        Ground plane: the list [a, b, c, d] such that for each point (x, y, z) we have ax + by + cz + d = 0
        """        

        self.best_plane = None 
        
        max_num_inliers = -1
        best_inliers = None
        
        num_points = len(self.points)
        
        # only need x, y,z information for estimation
        xyz = self.points[:, :3]
        
        for _ in range(self.num_iterations):
            # Random pick 3 samples from all points
            sample_indices = np.random.choice(num_points, 3, replace=False)
            sample_points = xyz[sample_indices]
            p1, p2, p3 = sample_points[:, :3]
            normal = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(normal) == 0:
                continue  # Skip if normal is zero to avoid dividing by zero
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, p1)
            plane_model = [normal[0], normal[1], normal[2], d]
            
            distances = np.abs(np.dot(xyz, normal) + d) / np.linalg.norm(normal)
            inliers = np.where(distances < self.distance_threshold)[0]

            if len(inliers) > max_num_inliers:
                max_num_inliers = len(inliers)
                self.best_plane = plane_model
                best_inliers = inliers   
        
        ground_points = np.take(self.points, best_inliers, axis=0)

        # # number of ground points
        self.num_points = len(ground_points)
        
        # # x field as shape (self.num_points, )
        self.pc_x = ground_points[:, 0]
        
        # # y field as shape (self.num_points, )
        self.pc_y = ground_points[:, 1]
        
        # # z field as shape (self.num_points, )
        self.pc_z = ground_points[:, 2]
        
        # # intensity field as shape (self.num_points, )
        self.pc_intensity = ground_points[:, 3]

    def get_index_inrange(self, start, end):
        """Get indices of pointcloud field "y" data that has value in range [start, end)
        
        Args:
            - start: start value
            - end: end value 
        """    
            
        indices = [i for i in range(self.num_points) if self.pc_y[i] >= start and self.pc_y[i] < end]
        return indices
        
    def build_intensity_histogram(self):
        """Build intensity histogram from 'y' field and 'intensity' field
        """        
        
        min_y = self.pc_y.min()
        max_y = self.pc_y.max()
        num_y_bins = math.ceil((max_y - min_y) / self.intensity_bin_res)

        intensity_vals = np.zeros((num_y_bins - 1))
        y_vals = np.zeros((num_y_bins - 1))
        
        # allocate "y" values into bins
        y_bins = np.linspace(min_y, max_y, num_y_bins)
        
        for i in range(num_y_bins - 1):
            indices = self.get_index_inrange(y_bins[i], y_bins[i + 1])
            intensity_sum = 0
            for index in indices:
                intensity_sum += self.pc_intensity[index]
            intensity_vals[i] = intensity_sum
            y_vals[i] = (y_bins[i] + y_bins[i + 1]) / 2
        
        
        return y_vals, intensity_vals
    
    def detect_peaks(self, y_vals, intensity_vals):
        """Find all intensity peaks and 2 "lane peaks" that correspond to the potential left and right lanes

        Args:
            - y_vals (List, float): y value bins
            - intensity_vals (List, float): intensity value bins

        Returns:
            - selected_ys (List, float): y value bins of the left and right lanes
            - selected_intensities (List, float): intensity value bins of the left and right lanes
            - y_peaks (List, float): y values at every intensity peak
            - intensity_peaks (List, float): intensity value at every intensity peak
        """  
           
        # Find peaks inside a signal: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
        peak_indices = find_peaks(intensity_vals)[0]
   
        y_peaks = y_vals[peak_indices]
        intensity_peaks = intensity_vals[peak_indices]
        
        # based on assumption that the left lane belongs to the left half, wheras
        # the right lane belongs to the right half (with respect to the ego vehicle position as the center)
        left_lane_indices = y_peaks >= 0
        right_lane_indices = y_peaks < 0
        
        left_lane_ys = y_peaks[left_lane_indices]
        right_lane_ys = y_peaks[right_lane_indices]
                
        left_lane_intensities = intensity_peaks[left_lane_indices]
        right_lane_intensities = intensity_peaks[right_lane_indices]
        
        row = -1
        col = -1
        min_dif = np.inf
        
        for i, left_y in enumerate(left_lane_ys):
            
            cache_difs = np.abs(self.lane_width - (left_y - right_lane_ys))
            col_index = np.argmin(cache_difs)
            cache_dif = cache_difs[col_index]
            if cache_dif < min_dif:
                row = i
                col = col_index
        
        selected_ys = [left_lane_ys[row], right_lane_ys[col]]
        selected_intensities = [left_lane_intensities[row], right_lane_intensities[col]]
        
        # If the calculated lane width is not within the bound, return the lane with highest peak
        estimated_lane_width = left_lane_ys[row] - right_lane_ys[col]
        if abs(estimated_lane_width - self.lane_width) > self.lane_bound_threshold:
            max_left_index = np.argmax(left_lane_intensities)
            max_right_index = np.argmax(right_lane_intensities)
            selected_ys = [left_lane_ys[max_left_index], right_lane_ys[max_right_index]]
            selected_intensities = [left_lane_intensities[max_left_index], right_lane_intensities[max_right_index]]

        return selected_ys, selected_intensities, y_peaks, intensity_peaks         

    def plot_peaks(self, y_accum, intensity_accum, y_peaks, intensity_peaks, selected_ys, selected_intensities):
        """Plot peak detection results

        Args:
            y_accum (List, float): y values in intensity histogram
            intensity_accum (List, float): intensity values in intensity histogram
            y_peaks (List, float): y values in every detected peak
            intensity_peaks (List, float): intensity values in every detected peak
            selected_ys (List, float): y values in lane peaks
            selected_intensities (List, float): intensity values in lane peaks
        """    
            
        # Save intensity peaks figure
        plt.figure()
        plt.title('Intensity peak detection')
        plt.plot(y_accum, intensity_accum,'--k', label='Histogram')
        plt.plot(y_peaks, intensity_peaks, '*', label='Peaks')
        plt.plot(selected_ys, selected_intensities,'o', label='Lane peaks') 
        plt.xlabel('y')
        plt.ylabel('intensity')
        plt.legend()
        peak_figure_path = f"{self.peak_figure_dir}/{self.basename}.jpg"
        plt.savefig(peak_figure_path)
        plt.close()
                
    def detect_lanes(self, lane_start_ys):
        """Detect `x` + `y` field of lane points using sliding window approach

        Args:
            - lane_start_ys: "start" y values obtained from the peak detection step
        """      
        min_x = self.pc_x.min()
        max_x = self.pc_x.max()
        self.num_x_bins = math.ceil(max_x - min_x)
        
        x_bins = np.linspace(min_x, max_x, self.num_x_bins)
        
        vertical_bins = np.zeros((self.num_x_bins - 1, 2, self.num_lanes)) #only use x & y
        
        lanes = np.zeros((self.num_x_bins - 1, 2, self.num_lanes)) #only use x & y
                
        for i in range(self.num_x_bins - 1):
            for j in range(self.num_lanes):                
                indices = np.where((self.pc_x < x_bins[i + 1]) 
                                   & (self.pc_x >= x_bins[i])  
                                   & (self.pc_y < lane_start_ys[j] + self.horizontal_bin_res / 2)
                                   & (self.pc_y >= lane_start_ys[j] - self.horizontal_bin_res / 2))[0]

                if len(indices) != 0:
                    
                    # retrieve pointcloud data inside ROI
                    roi_accum_intensity = self.pc_intensity[indices]
                    roi_accum_x = self.pc_x[indices]
                    roi_accum_y = self.pc_y[indices]
                    
                    max_intensity_index = np.argmax(roi_accum_intensity)

                    vertical_bins[i, 0, j] = roi_accum_x[max_intensity_index]
                    vertical_bins[i, 1, j] = roi_accum_y[max_intensity_index]
                    
                    lanes[i, :, j] = vertical_bins[i, :, j]
                    lane_start_ys[j] = vertical_bins[i, 1, j]
                else:
                    value = lanes[1:, :, j]
                    value = value[~np.all(value == 0, axis=1)]

                    if value.shape[0] == 2:  # Linear prediction (2 samples)
                        try:
                            P = fit_polynomial_ransac(value, 1, 0.1)
                        except:
                            continue
                    elif value.shape[0] > 2:  # 2-degree polynomial prediction (more than 2 samples)
                        try:
                            P = fit_polynomial_ransac(value, 2, 0.1)
                        except:
                            continue
                    else:
                        vertical_bins[i, :, j] = vertical_bins[-1, :, j]
                        continue
                    
                    error = np.mean(np.sqrt((np.polyval(P, value[:, 0]) - value[:, 1]) ** 2))
                    if error < 0.1:
                        xval = (x_bins[i] + x_bins[i + 1]) / 2
                        yval = np.polyval(P, xval)
                        yval -= error * abs(yval)
                        lane_start_ys[j] = yval
                        
                        vertical_bins[i, :, j] = [xval, yval]
                         
        return lanes
    
    def fit_polynomial(self, lanes):
        """Estimate a polynomial fitting from the detected left and right lanes

        Args:
            - lanes: detected left and right lanes

        Returns:
            - coeffs (List[List], float): polynomial coefficients of the detected lanes
            - errors(List, float): the corresponding mean square errors of the polynomial fittings
            - xs (List[List], float): accumulated x on the detected lanes
            - ys (List[List], float): accumulated y on the detected lanes
        """        
        coeffs = []
        errors = []
        xs = []
        ys = []
        
        for i in range(self.num_lanes):
            lane = [lanes[j, :, i] for j in range(self.num_x_bins - 1) if lanes[j, 0, i] != 0 and lanes[j, 1, i] != 0]
            
            lane = np.array(lane).reshape(-1, 2)
            
            x = lane[:, 0] # x values
            y = lane[:, 1] # y values
        
            if len(x) > 0 and len(y) > 0:
                
                # need to reverse because poly.polyfit returns the coefficients from low to high degree
                coef = poly.polyfit(x, y, self.degree)[::-1]
                pts_square = (np.polyval(coef, x) - y)**2
                error = np.sqrt(np.mean(pts_square))
            else:
                coef = [0, 0, 0, 0]
                error = np.inf
            
            print(f"lane {i} coefficients:", coef)
            print(f"lane {i} mean square error:", error)
            print()
            
            coeffs.append(coef)
            errors.append(error)
            
            xs.append(x)
            ys.append(y)

        return coeffs, errors, xs, ys  

    def plot_lanes(self, coefs, xs):
        """Plot detected lanes

        Args:
            - coefs (List[List], float): polynomial coefficients estimated from fit_polynomial
            - xs (List[List], float): x field of detected lane points
        """        
        plt.figure()
        plt.title('Detected lane points')
        plt.plot(self.pc_x, self.pc_y)
        plt.plot(xs[0], np.polyval(coefs[0], xs[0]), '*y', label='Left Lane')
        plt.plot(xs[1], np.polyval(coefs[1], xs[1]), '*r', label='Right Lane')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        lane_figure_path = f"{self.lane_figure_dir}/{self.basename}.jpg"
        plt.savefig(lane_figure_path)
        plt.close()        
    
    def save_coefficients(self, coefs):
        """Save estimated polynomial coefficients into text file, following the assignment required format

        Args:
            - coefs (List[List], float): polynomial coefficients estimated from fit_polynomial
        """        
        output_path = f"{self.output_dir}/{self.basename}.txt"
        with open(output_path, "w+") as f:
            for coef in coefs:
                n = len(coef)
                for j in range(n):
                    f.write(f"{coef[j]}")
                    if j < n - 1:
                        f.write(";")
                    else:
                        f.write("\n")
                           
    def apply_parallel_fitting(self, coefs, errs, xs):
        """Apply parallel constraint fitting to refine the estimated polynomial coefficients

        Args:
            - coefs (List[List], float): polynomial coefficients estimated from fit_polynomial
            - errs (List, float): mean square errors of each polynomial coefficients
            - xs (List[List], float): accumulated x on the detected lanes

        Returns:
            - "refined" coefs
        """        
        yval1 = np.polyval(coefs[0], xs[0])
        yval2 = np.polyval(coefs[1], xs[1])   
        
        zval1 = -(self.best_plane[0] * xs[0] - self.best_plane[1] * yval1 - self.best_plane[3]) / self.best_plane[2]
        zval2 = -(self.best_plane[0] * xs[1] - self.best_plane[1] * yval2 - self.best_plane[3]) / self.best_plane[2]

        lane3d1 = np.column_stack((xs[0], yval1, zval1))
        lane3d2 = np.column_stack((xs[1], yval2, zval2))
        
        if errs[0] > errs[1]:
            cache_coef = copy.deepcopy(coefs[1])
            if lane3d1[0, 1] > 0:
                cache_coef[3] = lane3d2[0, 1] + self.lane_width
            else:
                cache_coef[3] = lane3d2[0, 1] - self.lane_width
            coefs[0] = cache_coef
        else:
            cache_coef = copy.deepcopy(coefs[0])
            if lane3d2[0, 1] > 0:
                cache_coef[3] = lane3d1[0, 1] + self.lane_width
            else:
                cache_coef[3] = lane3d1[0, 1] - self.lane_width
            coefs[1] = cache_coef  
            
        return coefs

    
    def pipeline(self, input_path):
        """End-to-end pipeline to detect the left and right lane lines
        and estimate their polynomial coefficients from the input binary file

        Args:
            - input_path (str): input binary file path
        """            
        
        # get bin file basename (without '.bin' extension)
        self.basename = os.path.basename(input_path).split('.')[0]
        
        # Read lidar pointcloud data
        self.read_pointcloud_data(input_path)
        
        # Step 1. Extract Ground Points
        self.extract_ground_points()
        
        # Step 2. Build Intensity Histogram
        y_accum, intensity_accum = self.build_intensity_histogram()
        
        # Step 3. Detect Intensity Peaks
        selected_ys, selected_intensities, y_peaks, intensity_peaks  = self.detect_peaks(y_accum, intensity_accum)

        # Step 3-extra. Plot peak detection result
        self.plot_peaks(y_accum, intensity_accum, y_peaks, intensity_peaks, selected_ys, selected_intensities)
        
        # Step 4. Detect the possible lane points based on the sliding window approach
        lanes = self.detect_lanes(selected_ys)
        
        # Step 5. Estimate a polynomial curve on the detected lane points
        coefs, _, xs, _ = self.fit_polynomial(lanes)

        # Step 6. optional
        # coefs = self.apply_parallel_fitting(coefs, errs, xs)
                       
        # Step 5-Extra 1. Save detected lane figure
        self.plot_lanes(coefs, xs)
        
        # Step 5-Extra 2. Save lane polynomial coefficients
        self.save_coefficients(coefs)

