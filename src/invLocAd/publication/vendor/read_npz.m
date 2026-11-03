function data = read_npz(path)
% Read the numeric arrays of a NumPy .npz archive into a struct.
% Inputs:  path - .npz file written by numpy.savez with numeric arrays
% Outputs: data - struct with one field per array, named as in the archive
%                 (characters other than letters, digits and underscores become underscores);
%                 1-D arrays are column vectors and N-D arrays keep the NumPy shape
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder, 's'));
names = unzip(path, folder);
data = struct();
for i = 1:numel(names)
    [~, name] = fileparts(names{i});
    data.(matlab.lang.makeValidName(name, 'ReplacementStyle', 'underscore')) = read_npy(names{i});
end
end

function values = read_npy(path)
% Read one little-endian numeric .npy array (format versions 1-3).
stream = fopen(path, 'r', 'ieee-le');
closer = onCleanup(@() fclose(stream));
magic = fread(stream, 6, 'uint8=>uint8')';
if ~isequal(magic, [147 uint8('NUMPY')])
    error('read_npz:Format', 'Not a NumPy array file: %s', path);
end
version = fread(stream, 2, 'uint8');
if version(1) == 1
    header_length = fread(stream, 1, 'uint16');
else
    header_length = fread(stream, 1, 'uint32');
end
header = fread(stream, [1 header_length], 'char=>char');
descr = regexp(header, '''descr'':\s*''([^'']+)''', 'tokens', 'once');
fortran = contains(header, '''fortran_order'': True');
shape_text = regexp(header, '''shape'':\s*\(([^)]*)\)', 'tokens', 'once');
shape = str2double(strsplit(strtrim(shape_text{1}), ','));
shape = shape(~isnan(shape));
types = struct('f8', 'double', 'f4', 'single', 'i8', 'int64', 'i4', 'int32', ...
    'u8', 'uint64', 'u4', 'uint32', 'b1', 'uint8', 'u1', 'uint8', 'i1', 'int8');
code = descr{1}(2:end);
if ~any(descr{1}(1) == '<|') || ~isfield(types, code)
    error('read_npz:Type', 'Unsupported array type %s in %s', descr{1}, path);
end
count = prod(shape);
values = fread(stream, count, [types.(code) '=>' types.(code)]);
if isempty(shape)
    return
elseif isscalar(shape)
    values = reshape(values, shape, 1);
elseif fortran
    values = reshape(values, shape);
else
    values = permute(reshape(values, fliplr(shape)), numel(shape):-1:1);
end
end
