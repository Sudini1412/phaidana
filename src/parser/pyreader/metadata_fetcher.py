import midas.file_reader as fr 

# Open our file
mfile = fr.MidasFile("/bundle/data/DarkSide/phaidaq/run02038/run02038sub0000.mid")

# We can simply iterate over all events in the file
# for event in mfile:
#     bank_names = ", ".join(b.name for b in event.banks.values())
#     print("Event # %s of type ID %s contains banks %s" % (event.header.serial_number, event.header.event_id, bank_names))
   
odb = mfile.get_bor_odb_dump().data   

###### Run metadata
run_number = odb["Runinfo"]["Run number"]
metadata = odb["Experiment"]["Edit on Start"]

#print(f'This run is run{run_number} and was taken by {metadata["shifter"]}')
#print(f'This run was {metadata["run type"]} for {metadata["comment"]}')

event = mfile.read_this_event_body()
print(event.header.event_id)
