# Paths to preproccesed file Tomas

These paths are done in the pyspark, so the hdfs folder is under user=ebouille

Array :
All arrays are saved in .txt files

### Stop_times
Path:


Structure: [[arrival_time, departure time],[],...]
Ordered by: route_id, trip_id, stop_id

Table with the same order
'data/lgpt_guys/stop_times_ordered_like_array.csv'

### Routes


Structure
[[n_Trips, n_stops, Cumsum_trips ("pointer like function"), Cumsum_stops("pointer like function")]]
Ordered by: "route_id"

### Route stops
Path:


Structure: [[route_int1], [route_int2],....]
(Note: the functions to_numpy() always put brackets between elements of each row, this is the reason the structure is not [route_int1, route_int2,....] )

Ordered by : "route_id", "stop_sequence"

### Stops
Path:
'data/lgpt_guys/stops_array_version3.txt'

Structure: [[Cumsu transfer ("pointer like function"), cumsum routes ("pointer like function")]]

Ordered by: stop_id

### Stop_routes
Path:


Structure: [[route_int], [route_int],....]

Ordered by : "stop_id", "route_id"

### Transfer
Path:


Structure: [[stop_int2, Transfer_time_sec],....]

Ordered by : "stop_id"