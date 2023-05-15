
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, Polygon
from shapely.affinity import rotate
import xml.etree.ElementTree as ET



def read_rcorthobox(diclist):
    boxlist = []
    #print(series[rcorthofield])
    for item in diclist:
        if Path(item['orthoboxfile']).is_file:
            #print(item)s
            with open(Path(item['orthoboxfile']), 'r') as f:
                contents = f.read()
            xmls = contents.split('</OrthoProjection>')
            ortho_xml = xmls[0] + '</OrthoProjection>'
            recon_xml = xmls[1].lstrip('<')
            #print(recon_xml)
            # Read the reconstruction region box coordinates from the second xml
            reconstruction_region = ET.fromstring(recon_xml)
            try:   
                x,y,z = tuple(map(float,reconstruction_region.find('CentreEuclid').attrib['centre'].split()))
            except:
                x,y,z = tuple(map(float, reconstruction_region.find('CentreEuclid').find('centre').text.split()))
            #print(center_elem )
            #center_point = tuple(map(float, center_elem.split()))
            try:
                width, height, depth = tuple(map(float, reconstruction_region.attrib['widthHeightDepth'].split()))
            except:
                width, height, depth = tuple(map(float, reconstruction_region.find('widthHeightDepth').text.split()))
            #print(width, height, depth )
            # Create the 3D box geometry as a Shapely Polygon
            half_width = width / 2
            half_height = height / 2
            half_depth = depth / 2
            coordinates = [
                (x - half_width, y - half_height),
                (x - half_width, y + half_height),
                (x + half_width, y + half_height),
                (x + half_width, y - half_height)
            ]
            box1 = Polygon(coordinates)

            # Rotate the box to match the yawPitchRoll rotation in the XML file
            try:
                yaw, pitch, roll = tuple(map(float, reconstruction_region.attrib['yawPitchRoll'].split()))
            except:
                yaw, pitch, roll = tuple(map(float, reconstruction_region.find('yawPitchRoll').text.split()))
            print(yaw,pitch,roll)
            box_3d = rotate(box1, 180 - roll, origin=(x,y))  # Rotate around the z-axis
            print(box_3d)
            box1 = {}
            box1['geometry'] = box_3d
            box1['name'] = Path(item['orthoboxfile']).stem
            box1['orthoprojection'] = ortho_xml
            boxlist.append(box1.copy())
    # Create a GeoDataFrame with the box geometry
    orthobox_gpdf = gpd.GeoDataFrame(boxlist, geometry='geometry')

    # Add CRS information if available
    if len(boxlist) > 0 and 'globalCoordinateSystem' in reconstruction_region.attrib:
        crs = reconstruction_region.attrib['globalCoordinateSystem']
        orthobox_gpdf.crs = crs

    return orthobox_gpdf
