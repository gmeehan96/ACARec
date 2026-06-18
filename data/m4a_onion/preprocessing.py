import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


def get_dates(TEST_DATE, train_years=1, cold_val_months=1, test_months=1):
    test_date = pd.to_datetime(TEST_DATE)
    train_date = test_date - pd.DateOffset(years=train_years)
    cold_val_date = test_date - pd.DateOffset(months=cold_val_months)
    end_date = test_date + pd.DateOffset(months=test_months)
    return train_date, cold_val_date, test_date, end_date


compress = (
    lambda x: x.groupby(["user_id", "track_id"])
    .agg(timestamp=("timestamp", "min"), count=("timestamp", "count"))
    .reset_index()
)


def date_split(
    df,
    TEST_DATE,
    mapping,
    train_years=1,
    test_months=3,
    cold_val_months=1,
    min_items_per_user=1,
    min_users_per_item=1,
):
    train_date, cold_val_date, test_date, end_date = get_dates(
        TEST_DATE, train_years, cold_val_months, test_months
    )

    # Select time span
    df = df[(df.timestamp >= train_date) & (df.timestamp < end_date)]

    track_mins = df.groupby("track_id")["timestamp"].min().reset_index()
    cold_val_tracks = set(
        track_mins[
            (track_mins.timestamp >= cold_val_date) & (track_mins.timestamp < test_date)
        ]["track_id"].values
    )

    # Get test data
    te = compress(df)
    te = te[te.timestamp >= test_date]
    te = te.merge(mapping.loc[:, "artist"], left_on="track_id", right_index=True)
    cold_val_te = te[te.track_id.isin(cold_val_tracks)]
    te = te[~te.track_id.isin(cold_val_tracks)]

    # This is done so that the number of plays in train doesn't account for plays that happened after TEST_DATE
    df = df[df.timestamp < test_date]
    tr = compress(df)
    tr = tr.merge(mapping.loc[:, "artist"], left_on="track_id", right_index=True)
    cold_val_tr = tr[tr.track_id.isin(cold_val_tracks)]
    tr = tr[~tr.track_id.isin(cold_val_tracks)]

    cold_val = pd.concat([cold_val_tr, cold_val_te])

    # Track items before filtering
    items_before_filtering = tr.track_id.unique()

    # Apply filtering: loop until both constraints are satisfied, do filtering if needed—without using while True
    needs_filtering = True
    while needs_filtering:
        needs_filtering = False

        if min_users_per_item > 1:
            item_counts = tr.groupby("track_id")["user_id"].count()
            if item_counts.min() < min_users_per_item:
                valid_items = item_counts[item_counts >= min_users_per_item].index
                tr = tr[tr.track_id.isin(valid_items)]
                needs_filtering = True

        if min_items_per_user > 1:
            user_counts = tr.groupby("user_id")["track_id"].count()
            if user_counts.min() < min_items_per_user:
                valid_users = user_counts[user_counts >= min_items_per_user].index
                tr = tr[tr.user_id.isin(valid_users)]
                needs_filtering = True

    # Identify items that became cold due to filtering
    items_after_filtering = tr.track_id.unique()
    filtered_out_items = np.setdiff1d(items_before_filtering, items_after_filtering)

    # Filter cold users out
    te = te[te.user_id.isin(tr.user_id.unique())]

    # Split test into hot and cold parts
    test_item_isin_train = te.track_id.isin(tr.track_id.unique())
    cold = te[~test_item_isin_train]
    hot = te[test_item_isin_train]

    # Extract interactions with filtered items into cold_val
    # cold_val = cold[cold.track_id.isin(filtered_out_items)]
    cold_val = cold_val[cold_val.user_id.isin(tr.user_id.unique())]
    cold = cold[~cold.track_id.isin(filtered_out_items)]

    # We don't want to put users with cold items in val because there's so little of them
    users_without_cold_items = hot[
        ~hot.user_id.isin(cold.user_id.unique())
    ].user_id.unique()

    # Split users into val and test
    validation_user_ids, test_user_ids = train_test_split(
        users_without_cold_items, test_size=0.7, random_state=42
    )
    val = hot[hot.user_id.isin(validation_user_ids)]
    test = hot[~hot.user_id.isin(validation_user_ids)]

    artist_cold = cold.loc[~cold.artist.isin(tr.artist.unique())]
    artist_hot = cold.loc[cold.artist.isin(tr.artist.unique())]

    # Split cold_val by artist presence in training
    cold_hot_val = cold_val.loc[cold_val.artist.isin(tr.artist.unique())]
    cold_cold_val = cold_val.loc[~cold_val.artist.isin(tr.artist.unique())]

    return tr, val, test, artist_hot, artist_cold, cold_hot_val, cold_cold_val


def encode_ids(
    train, val, hot_test, cold_hot_test, cold_cold_test, cold_hot_val, cold_cold_val
):
    all_users = train.user_id.unique()
    all_items = np.sort(
        np.concatenate(
            [
                train.track_id.unique(),
                cold_hot_test.track_id.unique(),
                cold_cold_test.track_id.unique(),
                cold_hot_val.track_id.unique(),
                cold_cold_val.track_id.unique(),
            ]
        )
    )

    all_artists = np.concatenate(
        [
            train.artist.unique(),
            cold_cold_test.artist.unique(),
            cold_cold_val.artist.unique(),
        ]
    )

    ue = LabelEncoder()
    ie = LabelEncoder()
    ae = LabelEncoder()

    ue.fit(all_users)
    ie.fit(all_items)
    ae.fit(all_artists)

    train["user_id"] = ue.transform(train["user_id"])
    train["item_id"] = ie.transform(train["track_id"])
    train["artist_id"] = ae.transform(train["artist"])
    val["user_id"] = ue.transform(val["user_id"])
    val["item_id"] = ie.transform(val["track_id"])
    val["artist_id"] = ae.transform(val["artist"])
    hot_test["user_id"] = ue.transform(hot_test["user_id"])
    hot_test["item_id"] = ie.transform(hot_test["track_id"])
    hot_test["artist_id"] = ae.transform(hot_test["artist"])
    cold_hot_test["user_id"] = ue.transform(cold_hot_test["user_id"])
    cold_hot_test["item_id"] = ie.transform(cold_hot_test["track_id"])
    cold_hot_test["artist_id"] = ae.transform(cold_hot_test["artist"])
    cold_cold_test["user_id"] = ue.transform(cold_cold_test["user_id"])
    cold_cold_test["item_id"] = ie.transform(cold_cold_test["track_id"])
    cold_cold_test["artist_id"] = ae.transform(cold_cold_test["artist"])

    if len(cold_hot_val) > 0:
        cold_hot_val["user_id"] = ue.transform(cold_hot_val["user_id"])
        cold_hot_val["item_id"] = ie.transform(cold_hot_val["track_id"])
        cold_hot_val["artist_id"] = ae.transform(cold_hot_val["artist"])

    if len(cold_cold_val) > 0:
        cold_cold_val["user_id"] = ue.transform(cold_cold_val["user_id"])
        cold_cold_val["item_id"] = ie.transform(cold_cold_val["track_id"])
        cold_cold_val["artist_id"] = ae.transform(cold_cold_val["artist"])

    train.drop("track_id", axis=1, inplace=True)
    val.drop("track_id", axis=1, inplace=True)
    hot_test.drop("track_id", axis=1, inplace=True)
    cold_hot_test.drop("track_id", axis=1, inplace=True)
    cold_cold_test.drop("track_id", axis=1, inplace=True)
    cold_hot_val.drop("track_id", axis=1, inplace=True)
    cold_cold_val.drop("track_id", axis=1, inplace=True)

    train.drop("artist", axis=1, inplace=True)
    val.drop("artist", axis=1, inplace=True)
    hot_test.drop("artist", axis=1, inplace=True)
    cold_hot_test.drop("artist", axis=1, inplace=True)
    cold_cold_test.drop("artist", axis=1, inplace=True)
    cold_hot_val.drop("artist", axis=1, inplace=True)
    cold_cold_val.drop("artist", axis=1, inplace=True)

    return (
        train,
        val,
        hot_test,
        cold_hot_test,
        cold_cold_test,
        cold_hot_val,
        cold_cold_val,
        ue,
        ie,
        ae,
    )


TIMESTAMP_DATA = ".../userid_trackid_timestamp.tsv"  # M4A-Onion interaction data
ID_INFORMATION = ".../id_information.csv"  # M4A metadata
TEST_DATE = "2018-09-01"
n_core = 5

# load and filter interaction data
print("Loading timestamp data...")
df = pd.read_csv(TIMESTAMP_DATA, sep="\t")
df["year"] = df["timestamp"].apply(lambda x: int(x.split("-")[0]))
df = df[df["year"].isin([2017, 2018])]
df["timestamp"] = df["timestamp"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
)

# remap track IDs
track_ids = pd.read_csv("trackid_sorted.csv", index_col=0)
track_ids_map = track_ids.reset_index().set_index("trackid").to_dict()["idx"]
df["track_id"] = df["track_id"].apply(lambda x: track_ids_map[x])
mapping = pd.read_csv(ID_INFORMATION, sep="\t")
mapping = track_ids.merge(mapping, left_on="trackid", right_on="id").drop("id", axis=1)

print("Splitting data...")
# split by date
train, val, hot_test, cold_hot_test, cold_cold_test, cold_hot_val, cold_cold_val = (
    date_split(
        df, TEST_DATE, mapping, min_items_per_user=n_core, min_users_per_item=n_core
    )
)
(
    train,
    val,
    hot_test,
    cold_hot_test,
    cold_cold_test,
    cold_hot_val,
    cold_cold_val,
    ue,
    ie,
    ae,
) = encode_ids(
    train, val, hot_test, cold_hot_test, cold_cold_test, cold_hot_val, cold_cold_val
)

print("Saving outputs...")
# get artist_mapping
item_artist_df = (
    pd.concat(
        [
            train[["item_id", "artist_id"]].drop_duplicates(),
            cold_hot_test[["item_id", "artist_id"]].drop_duplicates(),
            cold_cold_test[["item_id", "artist_id"]].drop_duplicates(),
            cold_hot_val[["item_id", "artist_id"]].drop_duplicates(),
            cold_cold_val[["item_id", "artist_id"]].drop_duplicates(),
        ]
    )
    .drop_duplicates()
    .sort_values("item_id")
    .set_index("item_id")
)

item_artist_mapping = item_artist_df["artist_id"].values
np.save("item_artist_mapping.npy", item_artist_mapping)

# save interaction data

output_path = "./cold_item"
if not os.path.exists(output_path):
    os.makedirs(output_path)
full_df = pd.concat([train, val, hot_test, cold_hot_val, cold_hot_test])
user_num = max(full_df["user_id"]) + 1
item_num = max(full_df["item_id"]) + 1
info_dict = {"user": user_num, "item": item_num}
info_dict_path = os.path.join(output_path, "n_user_item.pkl")
pickle.dump(info_dict, open(info_dict_path, "wb"))

# save warm data
df_warm_train = train.loc[:, ["user_id", "item_id"]].rename(
    {"user_id": "user", "item_id": "item"}, axis=1
)
df_warm_val = val.loc[:, ["user_id", "item_id"]].rename(
    {"user_id": "user", "item_id": "item"}, axis=1
)
df_warm_test = hot_test.loc[:, ["user_id", "item_id"]].rename(
    {"user_id": "user", "item_id": "item"}, axis=1
)

df_warm_train.to_csv(os.path.join(output_path, "warm_train.csv"), index=False)
df_warm_val.to_csv(os.path.join(output_path, "warm_val.csv"), index=False)
df_warm_test.to_csv(os.path.join(output_path, "warm_test.csv"), index=False)

# save cold data
df_cold_val = cold_hot_val.loc[:, ["user_id", "item_id"]].rename(
    {"user_id": "user", "item_id": "item"}, axis=1
)
df_cold_test = cold_hot_test.loc[:, ["user_id", "item_id"]].rename(
    {"user_id": "user", "item_id": "item"}, axis=1
)
df_cold = pd.concat([df_cold_val, df_cold_test])

cold_object = "item"
df_cold.to_csv(os.path.join(output_path, f"cold_{cold_object}.csv"), index=False)
df_cold_val.to_csv(
    os.path.join(output_path, f"cold_{cold_object}_val.csv"), index=False
)
df_cold_test.to_csv(
    os.path.join(output_path, f"cold_{cold_object}_test.csv"), index=False
)


# overall data
warm_object = "item"
overall_val_user_set = np.array(
    list(set(df_cold_val[warm_object]) & set(df_warm_val[warm_object])), dtype=np.int32
)
df_overall_val = pd.concat([df_cold_val, df_warm_val])
df_overall_val = df_overall_val[df_overall_val[warm_object].isin(overall_val_user_set)]

overall_test_user_set = np.array(
    list(set(df_cold_test[warm_object]) & set(df_warm_test[warm_object])),
    dtype=np.int32,
)
df_overall_test = pd.concat([df_cold_test, df_warm_test])
df_overall_test = df_overall_test[
    df_overall_test[warm_object].isin(overall_test_user_set)
]

df_overall_val.to_csv(os.path.join(output_path, "overall_val.csv"), index=False)
df_overall_test.to_csv(os.path.join(output_path, "overall_test.csv"), index=False)


# get statistics
n_user_item = pickle.load(open(os.path.join(output_path, "n_user_item.pkl"), "rb"))
user_num = n_user_item["user"]
item_num = n_user_item["item"]

print("Global user_num: {}  item_num: {}".format(user_num, item_num))

emb_user = np.array(list(set(df_warm_train["user"])), dtype=np.int32)
warm_val_user = np.array(list(set(df_warm_val["user"])), dtype=np.int32)
warm_test_user = np.array(list(set(df_warm_test["user"])), dtype=np.int32)
cold_val_user = np.array(list(set(df_cold_val["user"])), dtype=np.int32)
cold_test_user = np.array(list(set(df_cold_test["user"])), dtype=np.int32)
overall_val_user = np.array(list(overall_val_user_set), dtype=np.int32)
overall_test_user = np.array(list(overall_test_user_set), dtype=np.int32)

emb_item = np.array(list(set(df_warm_train["item"])), dtype=np.int32)
warm_val_item = np.array(list(set(df_warm_val["item"])), dtype=np.int32)
warm_test_item = np.array(list(set(df_warm_test["item"])), dtype=np.int32)
cold_val_item = np.array(list(set(df_cold_val["item"])), dtype=np.int32)
cold_test_item = np.array(list(set(df_cold_test["item"])), dtype=np.int32)

overall_val_item = np.array(list(set(df_overall_val["item"])), dtype=np.int32)
overall_test_item = np.array(list(set(df_overall_test["item"])), dtype=np.int32)

# Statistics
user_array = np.arange(user_num, dtype=np.int32)
item_array = np.arange(item_num, dtype=np.int32)
warm_user = np.array(list(set(df_warm_train["user"].tolist())), dtype=np.int32)
warm_item = np.array(list(set(df_warm_train["item"].tolist())), dtype=np.int32)
cold_user = np.array(list(set(df_cold["user"].tolist())), dtype=np.int32)
cold_item = np.array(list(set(df_cold["item"].tolist())), dtype=np.int32)

print("[warm] user: {}  item: {}".format(len(warm_user), len(warm_item)))
print("[cold] user: {}  item: {}".format(len(cold_user), len(cold_item)))

# Save results
para_dict = {}
para_dict["user_num"] = user_num
para_dict["item_num"] = item_num
para_dict["user_array"] = user_array
para_dict["item_array"] = item_array
para_dict["warm_user"] = warm_user
para_dict["warm_item"] = warm_item
para_dict["cold_user"] = cold_user
para_dict["cold_item"] = cold_item

para_dict["train_user"] = emb_user
para_dict["warm_val_user"] = warm_val_user
para_dict["warm_test_user"] = warm_test_user
para_dict["cold_val_user"] = cold_val_user
para_dict["cold_test_user"] = cold_test_user
para_dict["overall_val_user"] = overall_val_user
para_dict["overall_test_user"] = overall_test_user

para_dict["train_item"] = emb_item
para_dict["warm_val_item"] = warm_val_item
para_dict["warm_test_item"] = warm_test_item
para_dict["cold_val_item"] = cold_val_item
para_dict["cold_test_item"] = cold_test_item
para_dict["overall_val_item"] = overall_val_item
para_dict["overall_test_item"] = overall_test_item

dict_path = os.path.join(output_path, "info_dict.pkl")
pickle.dump(para_dict, open(dict_path, "wb"), protocol=4)
