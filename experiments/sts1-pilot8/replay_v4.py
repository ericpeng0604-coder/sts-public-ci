"""Strict historical replay, with an explicit exception for equivalent map inputs."""
import math

def recorded_index(source,ordinal,kind,index,floor,descs,scores):
    actual={'kind':kind,'index':index,'floor':floor}
    expected=source['journal'][ordinal] if ordinal<len(source['journal']) else None
    if actual==expected:return index,None
    detail={'seed':source['seed'],'ordinal':ordinal,'actual':actual,'expected':expected}
    if expected and kind=='map' and expected['kind']==kind and expected['floor']==floor:
        historical=expected['index']
        if 0<=index<len(descs) and 0<=historical<len(descs) and len(scores)==len(descs):
            a,b=float(scores[index]),float(scores[historical])
            if descs[index]==descs[historical] and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-5:
                return historical,{**detail,'reason':'equivalent_map_inputs_float32_tie','score_gap':abs(a-b)}
    raise RuntimeError('replay_journal_mismatch:'+repr(detail))
