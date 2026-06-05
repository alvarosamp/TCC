from __future__ import annotations
from pathlib import Path

def c_identifier(name: str) -> str:
    '''
    Transforma em um transformador c valido

    '''
    chars = []
    for char in name:
        if char.isalnum():
            chars.append(char)
        else:
            chars.append('_')

    ident = "".join(chars).strip('_')

    if not ident:
        ident = 'model'
    
    if ident[0].isdigit():
        ident = '_' + ident

    return ident

def bytes_to_c_array(data: bytes,
                     values_per_line: int = 12) -> str:
    '''
    Converte bytes em uma string formatada como um array C.
    '''
    values = [str(byte) for byte in data]
    lines = []

    for i in range(0, len(values), values_per_line):
        lines.append(', '.join(values[i:i + values_per_line]))

    return '{\n    ' + ',\n    '.join(lines) + '\n}'

def export_tflite_to_header(
        tflite_path: str | Path,
        header_path: str | Path,
        array_name: str,
) -> dict:
    """
    Exporta .tflite para .h

    Retornando metadados simples : path, size_bytes, array_name
    """
    tflite_path = Path(tflite_path)
    header_path = Path(header_path)

    if not tflite_path.is_file():
        raise FileNotFoundError(f"Arquivo .tflite não encontrado: {tflite_path}")
    header_path.parent.mkdir(parents=True, exist_ok=True)
    data = tflite_path.read_bytes()
    array_name = c_identifier(array_name)
    guard = c_identifier(header_path.name).upper() 
    c_array = bytes_to_c_array(data)

    text = f'''#ifndef {guard}
#define {guard}
#include <stdint.h>
alignas(16) const unsigned char {array_name}[] = {{c_array}};
const unsigned int {array_name}_len = {len(data)};
#endif // {guard}
'''
    header_path.write_text(text)
    return {
        'header_path': str(header_path),
        'tflite_path': str(tflite_path),
        'array_names': [array_name],
        'size_bytes': len(data),
        'size_kb': len(data) / 1024,
    } 