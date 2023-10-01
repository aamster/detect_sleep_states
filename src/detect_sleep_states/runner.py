import argschema


class DetectSleepStatesSchema(argschema.ArgSchema):
    data_path = argschema.fields.InputFile(required=True)
    meta_path = argschema.fields.InputFile(required=True)
