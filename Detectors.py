import numpy as np
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

## Task 1: Linearity and saturation

## List of FITS files to process
fits_files = [
    'exp1_image1.fits', 'exp1_image2.fits', 'exp3_image1.fits', 'exp3_image2.fits',
    'exp6_image1.fits', 'exp6_image2.fits', 'exp9_image1.fits', 'exp9_image2.fits',
    'exp12_image1.fits', 'exp12_image2.fits', 'exp15_image1.fits', 'exp15_image2.fits',
    'exp18_image1.fits', 'exp18_image2.fits', 'exp21_image1.fits', 'exp21_image2.fits',
    'exp23_image1.fits', 'exp23_image2.fits', 'exp25_image1.fits', 'exp25_image2.fits',
    'exp27_image1.fits', 'exp27_image2.fits', 'exp29_image1.fits', 'exp29_image2.fits'
]

## Subframe coordinates
y0 = 896
y1 = y0 + 50
x0 = 1004
x1 = x0 + 50

exposure_time = np.array([1,3,6,9,12,15,18,21,23,25,27,29])

## Initializing arrays for counts median values in the subframe
image1_medians = []
image2_medians = []
counts1 = []
counts2 = []

## Processing each FITS file
for fits_file in fits_files:
    hdu = fits.open(fits_file)
    image_data = hdu[0].data.astype(float)
    
    # Handle possible multi-dimensional data
    if len(image_data.shape) > 2:
        image_data = image_data[0]
    
    # Extract subframe
    subcounts = image_data[y0:y1, x0:x1]
    
    # Calculate median
    median_value = np.median(subcounts)
    
    # Store in appropriate array based on image1 or image2
    if 'image1' in fits_file:
        image1_medians.append(median_value)
        counts1.append(subcounts)
        #print(f"{fits_file}: Median value = {median_value} (added to image1 array)")
    elif 'image2' in fits_file:
        image2_medians.append(median_value)
        counts2.append(subcounts)
        #print(f"{fits_file}: Median value = {median_value} (added to image2 array)")
    
    # Close the FITS file
    hdu.close()

## Converting lists to numpy arrays
image1_counts = np.array(image1_medians)
image2_counts = np.array(image2_medians)

counts1 = np.array(counts1)
counts2 = np.array(counts2)

# Print the final arrays
#print("\nImage1 medians array:", image1_medians)
#print("Image2 medians array:", image2_medians)

## Fitting arrays
fit_exp = exposure_time[:8]
fit_counts = image1_counts[:8]

## Fitting a straight line using linear regress
slope, intercept, r_value, p_value, std_err = stats.linregress(fit_exp, fit_counts)

## Calculating fitted values
fit_line = slope * fit_exp + intercept

print(f"R-squared: {r_value**2:.4f}") ## Determining the R^2 value to see how well the data (before saturation occurs) can be fitted by a straight line

## Plotting
plt.plot(exposure_time, image1_counts, 'bo', label='Data')
plt.plot(fit_exp, fit_line, 'r-', label='Fit')
plt.xlabel("Exposure Time (s)")
plt.ylabel("Counts")
plt.legend()
plt.show()

## Task 2: Gain

diff_subframe = counts1 - counts2
#print(diff_subframe[:8])
std_diff_subframe = np.std(diff_subframe[:8], axis=(1, 2))
#print(std_diff_subframe)
std_subframe1 = std_diff_subframe/np.sqrt(2)
var_subframe1 = std_subframe1**2
#print(var_subframe1) 

counts_subframe1 = image1_counts[:8]
slope, intercept = np.polyfit(counts_subframe1, var_subframe1, 1) ## Using polyfit to fit a straight line

gain = 1 / slope ## Gain is inverse of slope

## Plotting
plt.plot(counts_subframe1, var_subframe1, 'ro', label='Data')
plt.plot(counts_subframe1, slope * np.array(counts_subframe1) + intercept, 'b-', label='Fit')
plt.xlabel('Counts')
plt.ylabel('Variance')
plt.legend()
plt.show()

print("Gain:", gain)
print("slope:", slope)