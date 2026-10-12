import pathlib
ROOT=pathlib.Path(__file__).parent
G7=ROOT/"g7.pt"
def full_state(gc,policy,descs):
    from state_v3 import native_state,RELIC_IDS,native_card
    s=native_state(gc,policy,'card',descs,0)
    s['schema']='sts1-card-public-state-v4'
    s['deck_details']=[{'card':native_card(c),'upgrade_count':int(c.upgrade_count),'misc':int(c.misc),'innate':bool(c.innate),'type':str(c.type),'rarity':str(c.rarity)} for c in gc.deck]
    s['relic_details']=[{'id':RELIC_IDS[int(r.id)],'data':int(r.data),'data_visibility':'raw_native_not_audited_for_feature_use'} for r in gc.relics]
    s['visible_map']=list(gc.get_map_nn());s['map_position']={'x':int(gc.cur_map_node_x),'y':int(gc.cur_map_node_y)}
    s['public_extra']=dict(gc.public_noncombat_extra)
    s['unavailable_public_fields']=['relic_counter_semantics_not_yet_audited']
    s['feature_use_requires_audit']=['relic_details.data']
    return s
