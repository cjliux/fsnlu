#coding: utf-8


def collect_named_entities(self, labels):
    named_entities = []
    start_offset, end_offset, ent_type = None, None, None
        
    for offset, token_tag in enumerate(labels):
        if token_tag == 'O':
            if ent_type is not None and start_offset is not None: 
                end_offset = offset - 1
                named_entities.append((ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None
        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset
        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):
            end_offset = offset - 1
            named_entities.append((ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    if ent_type and start_offset and end_offset is None:
        named_entities.append((ent_type, start_offset, len(labels)-1))
    return named_entities

