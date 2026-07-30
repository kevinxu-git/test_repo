# Extract rows and columns
rows = [entry for entry in table_objects if entry['label'] == 'table row']
columns = [entry for entry in table_objects if entry['label'] == 'table column']

# Sort rows and columns by their Y and X coordinates, respectively
rows.sort(key=lambda x: x['bbox'][1])
columns.sort(key=lambda x: x['bbox'][0])


def find_cell_coordinates(row, column):
    cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
    return cell_bbox


def row_overlaps_header(row, header_bbox, overlap_threshold=0.5):
    """
    Determine whether a row's bbox falls inside the detected header's bbox,
    based on vertical (Y-axis) overlap. Using overlap ratio instead of a
    strict containment check makes this robust to rows that straddle the
    header boundary by a few pixels.
    """
    row_top, row_bottom = row['bbox'][1], row['bbox'][3]
    header_top, header_bottom = header_bbox[1], header_bbox[3]

    intersection_top = max(row_top, header_top)
    intersection_bottom = min(row_bottom, header_bottom)
    intersection = max(0, intersection_bottom - intersection_top)

    row_height = row_bottom - row_top
    if row_height <= 0:
        return False

    return (intersection / row_height) >= overlap_threshold


def find_detected_header_rows(rows, detected_header, overlap_threshold=0.5):
    """
    Given the sorted list of row entries and a detected header
    (either a dict with a 'bbox' key, e.g. {'label': 'table column header',
    'bbox': [...]}, or a raw bbox list/tuple), return the subset of rows
    that lie within the header's bounding box.
    """
    header_bbox = detected_header['bbox'] if isinstance(detected_header, dict) else detected_header

    detected_header_rows = [
        row for row in rows
        if row_overlaps_header(row, header_bbox, overlap_threshold)
    ]

    # Keep header rows sorted top to bottom, same convention as cell_coordinates
    detected_header_rows.sort(key=lambda x: x['bbox'][1])

    return detected_header_rows


def merge_header_rows_by_column(detected_header_rows, columns):
    """
    Collapse multiple header rows into a single merged header row, one
    bbox per column. Each column's merged bbox spans from the topmost
    header row to the bottommost header row, so multi-line column titles
    (e.g. wrapped headers) end up as one cell instead of one cell per line.

    Returns a single entry with the same shape as an item in
    cell_coordinates: {'row': merged_row_bbox, 'cells': [...], 'cell_count': N}
    """
    if not detected_header_rows:
        return None

    header_top = min(row['bbox'][1] for row in detected_header_rows)
    header_bottom = max(row['bbox'][3] for row in detected_header_rows)

    merged_cells = []
    for column in columns:
        merged_bbox = [column['bbox'][0], header_top, column['bbox'][2], header_bottom]
        merged_cells.append({'column': column['bbox'], 'bbox': merged_bbox})

    # Sort cells by X coordinate, same convention as row_cells
    merged_cells.sort(key=lambda x: x['column'][0])

    merged_row_bbox = [
        min(c['bbox'][0] for c in merged_cells),
        header_top,
        max(c['bbox'][2] for c in merged_cells),
        header_bottom,
    ]

    return {'row': merged_row_bbox, 'cells': merged_cells, 'cell_count': len(merged_cells)}


# Generate cell coordinates and count cells in each row
cell_coordinates = []

for row in rows:
    row_cells = []
    for column in columns:
        cell_bbox = find_cell_coordinates(row, column)
        row_cells.append({'column': column['bbox'], 'bbox': cell_bbox})

    # Sort cells in the row by X coordinate
    row_cells.sort(key=lambda x: x['column'][0])

    # Append row information to cell_coordinates
    cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

# Sort rows from top to bottom
cell_coordinates.sort(key=lambda x: x['row'][1])

# Detect which rows belong to the header, using the detected_header input
detected_header_rows = find_detected_header_rows(rows, detected_header)

# Concat header rows on the same column into a single merged header cell per column
merged_header_row = merge_header_rows_by_column(detected_header_rows, columns)

# Drop header rows from cell_coordinates entirely — cell_coordinates
# should only contain body (non-header) rows
header_row_bboxes = {tuple(row['bbox']) for row in detected_header_rows}
cell_coordinates = [
    entry for entry in cell_coordinates
    if tuple(entry['row']) not in header_row_bboxes
]

return detected_header_rows, merged_header_row, cell_coordinates
