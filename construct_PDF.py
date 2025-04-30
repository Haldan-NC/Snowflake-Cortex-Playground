from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
import os
import re



sample_output_1 = """
To resolve detergent residue issues during the fine wash cycle, follow these steps:

1. **Check Detergent Type and Dosage**:
   - Ensure you're using the recommended type of detergent for fine fabrics.
   - Use the appropriate amount of detergent based on the load and the degree of soiling. For light to normal soiling, use a smaller amount.

2. **Inspect the Detergent Drawer**:
   - Remove the detergent drawer and check for any blockages in the compartments. Clean the drawer thoroughly to ensure proper detergent flow.

3. **Adjust Water Settings**:
   - Avoid adding extra water manually during the cycle. Trust the appliance’s automatic load adjustment function to optimize water usage.

4. **Run a Maintenance Wash**:
   - Occasionally run an empty wash cycle with drum cleaning additives to clear out any residual detergent buildup.

5. **Check for Foam and Residue**:
   - If heavy foam is building up, consider reducing the detergent amount in subsequent cycles with similar loads.

6. **Ensure Proper Appliance Setup**:
   - Confirm that the appliance is correctly aligned and secured to prevent vibrations that could affect the washing process.
   - IMAGE_ID: 292, PATH: .\Washer_Images\WGA1420SIN\doc_6_page_16_img_0.png 

7. **Rinse Cycle Optimization**:
   - If detergent residue is persistent, try activating an additional rinse cycle, if available, especially if you notice residue after regular washing.

8. **Review Spin Speed**:
   - Using the maximum spin speed can sometimes aid in better detergent dispersion. However, ensure it's suitable for the fabric type.

If the issue persists, consider consulting the appliance's troubleshooting section or contacting customer support for further assistance.
"""