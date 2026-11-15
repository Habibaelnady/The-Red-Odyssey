
"""
Mars Weather-Conditioned Event Generator - V3

Input:
    mars_weather_simulation_7_sols.csv

Outputs:
    mars_weather_dependent_randomized.csv
    mars_weather_events_dependent_randomized.csv

Main logic:
    1. Read the original weather dataset.
    2. Calculate a wind threshold for each location.
    3. Find continuous elevated-wind windows in each original Sol.
    4. Prioritize Sols marked by dust_storm_event = 1.
    5. Generate events within the available wind windows.
    6. Shuffle complete Sol profiles with their events attached.

The original input file is never modified.

This is a gameplay simulation based on MCD climatological data,
not a scientific forecast of individual Martian storms.
"""

from pathlib import Path

import argparse
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "mars_weather_simulation_7_sols.csv"

WEATHER_OUTPUT = "mars_weather_dependent_randomized.csv"
EVENTS_OUTPUT = "mars_weather_events_dependent_randomized.csv"

SECONDS_PER_SOL = 420.0
HOURS_PER_SOL = 24.0

MIN_EVENT_DURATION = 30.0
PLANNED_MIN_DURATION = 45.0
PLANNED_MAX_DURATION = 140.0

MAX_EVENTS_PER_SOL = 1

# A qualifying window is identified using the 80th percentile
# of the location's median 25-hour wind profile.
WIND_PERCENTILE = 80

# Probability of generating an event on a Sol without
# the original dust_storm_event flag.
NORMAL_SOL_EVENT_PROBABILITY = 0.25

# The existing source flag is prioritized.
FLAGGED_SOL_EVENT_PROBABILITY = 1.0

# Event gameplay effects.
EVENT_EFFECTS = {
    "strong_winds": {
        "wind_multiplier": 1.65,
        "visibility": 100,
        "solar_power": 15,
    },

    "local_dust": {
        "wind_multiplier": 1.15,
        "visibility": 65,
        "solar_power": 11,
    },

    "moderate_dust_storm": {
        "wind_multiplier": 1.35,
        "visibility": 35,
        "solar_power": 8,
    },

    "severe_dust_storm": {
        "wind_multiplier": 1.55,
        "visibility": 15,
        "solar_power": 5,
    },
}

REQUIRED_COLUMNS = [
    "location",
    "sol",
    "local_time_martian_hour",
    "temperature_simulated_k",
    "pressure_simulated_pa",
    "solar_flux_w_m2",
    "horizontal_wind_speed_m_s",
    "daily_mean_dust_optical_depth",
]


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find input file: {path.resolve()}"
        )

    df = pd.read_csv(path)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    numeric_columns = [
        "sol",
        "local_time_martian_hour",
        "temperature_simulated_k",
        "pressure_simulated_pa",
        "solar_flux_w_m2",
        "horizontal_wind_speed_m_s",
        "daily_mean_dust_optical_depth",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "The dataset contains missing numeric values."
        )

    if "dust_storm_event" not in df.columns:
        df["dust_storm_event"] = 0

    df["dust_storm_event"] = (
        pd.to_numeric(
            df["dust_storm_event"],
            errors="raise"
        )
        .fillna(0)
        .astype(int)
    )

    if not set(df["dust_storm_event"].unique()).issubset({0, 1}):
        raise ValueError(
            "dust_storm_event must contain only 0 or 1."
        )

    if (df["horizontal_wind_speed_m_s"] < 0).any():
        raise ValueError(
            "Wind speed cannot be negative."
        )

    return df


# ============================================================
# TIME CONVERSION
# ============================================================

def add_game_time(df):

    result = df.copy()

    result["game_time_seconds"] = (
        result["local_time_martian_hour"]
        / HOURS_PER_SOL
        * SECONDS_PER_SOL
    )

    return result


def seconds_to_local_hour(seconds):

    return (
        seconds
        / SECONDS_PER_SOL
        * HOURS_PER_SOL
    )


# ============================================================
# WIND INTERPOLATION
# ============================================================

def interpolate_wind(profile, game_seconds):

    hours = profile[
        "local_time_martian_hour"
    ].to_numpy(dtype=float)

    wind = profile[
        "horizontal_wind_speed_m_s"
    ].to_numpy(dtype=float)

    sample_seconds = (
        hours / HOURS_PER_SOL * SECONDS_PER_SOL
    )

    return float(
        np.interp(
            game_seconds,
            sample_seconds,
            wind
        )
    )


def interpolate_column(profile, hour, column):

    hours = profile[
        "local_time_martian_hour"
    ].to_numpy(dtype=float)

    values = profile[column].to_numpy(dtype=float)

    return float(
        np.interp(
            hour,
            hours,
            values
        )
    )


# ============================================================
# LOCATION-SPECIFIC WIND THRESHOLD
# ============================================================

def calculate_wind_threshold(location_df):

    median_profile = (
        location_df
        .groupby("local_time_martian_hour")[
            "horizontal_wind_speed_m_s"
        ]
        .median()
        .reset_index()
        .sort_values("local_time_martian_hour")
    )

    if len(median_profile) != 25:
        raise ValueError(
            "Each location must have 25 hourly samples "
            "(Martian hours 0 through 24)."
        )

    threshold = float(
        np.percentile(
            median_profile[
                "horizontal_wind_speed_m_s"
            ],
            WIND_PERCENTILE
        )
    )

    return threshold, median_profile


# ============================================================
# FIND CONTINUOUS ELEVATED-WIND WINDOWS
# ============================================================

def find_wind_windows(
    profile,
    threshold,
    resolution_seconds=1.0
):

    sample_times = np.arange(
        0,
        SECONDS_PER_SOL + resolution_seconds,
        resolution_seconds
    )

    sample_times = np.minimum(
        sample_times,
        SECONDS_PER_SOL
    )

    wind_values = np.array([
        interpolate_wind(profile, t)
        for t in sample_times
    ])

    qualifying = wind_values >= threshold

    windows = []
    start = None

    for i, valid in enumerate(qualifying):

        if valid and start is None:
            start = float(sample_times[i])

        elif not valid and start is not None:

            end = float(sample_times[i - 1])

            if end > start:
                windows.append((start, end))

            start = None

    if start is not None:
        windows.append(
            (start, float(sample_times[-1]))
        )

    return windows


# ============================================================
# EVENT TYPE
# ============================================================

def determine_event_type(
    peak_wind,
    wind_threshold,
    source_dust_flag,
    location_thresholds,
):
    """
    For Sols already flagged as dust-storm days, classify
    the gameplay event according to the peak wind tier.

    For other Sols, elevated wind produces a strong-wind event.

    Percentile tiers are relative to the local climatology;
    they are not official Martian storm severity categories.
    """

    if source_dust_flag == 0:
        return "strong_winds"

    moderate_threshold = float(
        np.percentile(
            location_thresholds,
            92
        )
    )

    severe_threshold = float(
        np.percentile(
            location_thresholds,
            97
        )
    )

    if peak_wind >= severe_threshold:
        return "severe_dust_storm"

    if peak_wind >= moderate_threshold:
        return "moderate_dust_storm"

    if peak_wind >= wind_threshold:
        return "local_dust"

    return "strong_winds"


# ============================================================
# GENERATE EVENTS FOR ONE SOL
# ============================================================

def generate_sol_events(
    location,
    source_sol,
    sol_profile,
    wind_threshold,
    location_thresholds,
    rng,
    event_counter,
):

    sol_profile = sol_profile.sort_values(
        "local_time_martian_hour"
    ).copy()

    source_dust_flag = int(
        sol_profile["dust_storm_event"].max()
    )

    windows = find_wind_windows(
        sol_profile,
        wind_threshold
    )

    # Ignore windows too short to support a useful event.
    windows = [
        window
        for window in windows
        if (
            window[1] - window[0]
            >= MIN_EVENT_DURATION
        )
    ]

    if not windows:
        print(
            f"  Original Sol {source_sol}: "
            "no qualifying wind window"
        )

        return [], event_counter

    # This dataset normally has one main elevated-wind period.
    # If there are several, select one randomly.
    selected_window = windows[
        int(rng.integers(0, len(windows)))
    ]

    window_start, window_end = selected_window

    available_duration = (
        window_end - window_start
    )

    if available_duration < MIN_EVENT_DURATION:
        return [], event_counter

    probability = (
        FLAGGED_SOL_EVENT_PROBABILITY
        if source_dust_flag == 1
        else NORMAL_SOL_EVENT_PROBABILITY
    )

    if rng.random() > probability:

        print(
            f"  Original Sol {source_sol}: "
            "qualifying wind window, no event selected"
        )

        return [], event_counter

    # Start close to the beginning of the wind window.
    maximum_start_offset = min(
        5.0,
        available_duration - MIN_EVENT_DURATION
    )

    start_offset = float(
        rng.uniform(
            0,
            max(0, maximum_start_offset)
        )
    )

    start_seconds = window_start + start_offset

    remaining_window = (
        window_end - start_seconds
    )

    planned_duration = float(
        rng.uniform(
            PLANNED_MIN_DURATION,
            PLANNED_MAX_DURATION
        )
    )

    # The event cannot outlast the qualifying wind window.
    actual_duration = min(
        planned_duration,
        remaining_window
    )

    if actual_duration < MIN_EVENT_DURATION:
        return [], event_counter

    end_seconds = start_seconds + actual_duration

    # Check wind support throughout the event.
    verification_times = np.arange(
        start_seconds,
        end_seconds + 0.5,
        0.5
    )

    verification_winds = [
        interpolate_wind(sol_profile, t)
        for t in verification_times
    ]

    if not all(
        wind >= wind_threshold
        for wind in verification_winds
    ):
        print(
            f"  Original Sol {source_sol}: "
            "event rejected by wind-support validation"
        )

        return [], event_counter

    peak_wind = max(
        interpolate_wind(sol_profile, t)
        for t in np.linspace(
            window_start,
            window_end,
            100
        )
    )

    event_type = determine_event_type(
        peak_wind,
        wind_threshold,
        source_dust_flag,
        location_thresholds,
    )

    effects = EVENT_EFFECTS[event_type]

    intensity = float(
        rng.uniform(0.65, 1.0)
    )

    trigger_hour = seconds_to_local_hour(
        start_seconds
    )

    trigger_temperature = interpolate_column(
        sol_profile,
        trigger_hour,
        "temperature_simulated_k"
    )

    trigger_pressure = interpolate_column(
        sol_profile,
        trigger_hour,
        "pressure_simulated_pa"
    )

    trigger_solar = interpolate_column(
        sol_profile,
        trigger_hour,
        "solar_flux_w_m2"
    )

    trigger_wind = interpolate_wind(
        sol_profile,
        start_seconds
    )

    trigger_dust = interpolate_column(
        sol_profile,
        trigger_hour,
        "daily_mean_dust_optical_depth"
    )

    event_id = f"EVENT_{event_counter:03d}"

    event = {
        "event_id": event_id,
        "location": location,
        "source_sol": int(source_sol),
        "sol": None,
        "event_type": event_type,

        "start_time_seconds": round(
            start_seconds, 2
        ),

        "duration_seconds": round(
            actual_duration, 2
        ),

        "end_time_seconds": round(
            end_seconds, 2
        ),

        "intensity": round(
            intensity, 2
        ),

        "max_wind_multiplier": effects[
            "wind_multiplier"
        ],

        "minimum_visibility_percent": effects[
            "visibility"
        ],

        "solar_power": effects[
            "solar_power"
        ],

        "trigger_local_hour": round(
            trigger_hour, 3
        ),

        "trigger_temperature_k": round(
            trigger_temperature, 3
        ),

        "trigger_pressure_pa": round(
            trigger_pressure, 3
        ),

        "trigger_solar_flux_w_m2": round(
            trigger_solar, 3
        ),

        "trigger_wind_speed_m_s": round(
            trigger_wind, 3
        ),

        "peak_wind_speed_m_s": round(
            peak_wind, 3
        ),

        "wind_support_threshold_m_s": round(
            wind_threshold, 3
        ),

        "trigger_dust_optical_depth": round(
            trigger_dust, 6
        ),

        "source_dust_storm_event": source_dust_flag,
    }

    print(
        f"  Original Sol {source_sol}: "
        f"{event_type}, "
        f"{actual_duration:.1f} seconds"
    )

    return [event], event_counter + 1


# ============================================================
# SHUFFLE COMPLETE SOLS
# ============================================================

def shuffle_complete_sols(
    original_df,
    all_events,
    rng,
):

    weather_profiles = []
    randomized_events = []

    for location in sorted(
        original_df["location"].unique()
    ):

        location_df = original_df[
            original_df["location"] == location
        ].copy()

        original_sols = sorted(
            location_df["sol"].unique()
        )

        shuffled_sols = list(original_sols)
        rng.shuffle(shuffled_sols)

        print(
            f"\nRandomized Sol mapping: {location}"
        )

        for new_sol, source_sol in enumerate(
            shuffled_sols,
            start=1
        ):

            print(
                f"  Original Sol {source_sol} "
                f"-> New Sol {new_sol}"
            )

            profile = location_df[
                location_df["sol"] == source_sol
            ].copy()

            profile["source_sol"] = int(source_sol)
            profile["sol"] = int(new_sol)

            profile["event_type"] = ""
            profile["event_id"] = ""
            profile["event_start_seconds"] = np.nan
            profile["event_duration_seconds"] = np.nan
            profile["event_intensity"] = np.nan

            profile_events = [
                event.copy()
                for event in all_events
                if (
                    event["location"] == location
                    and event["source_sol"] == source_sol
                )
            ]

            for event in profile_events:

                event["sol"] = int(new_sol)

                trigger_hour = event[
                    "trigger_local_hour"
                ]

                nearest_index = (
                    profile[
                        "local_time_martian_hour"
                    ]
                    .sub(trigger_hour)
                    .abs()
                    .idxmin()
                )

                profile.loc[
                    nearest_index, "event_type"
                ] = event["event_type"]

                profile.loc[
                    nearest_index, "event_id"
                ] = event["event_id"]

                profile.loc[
                    nearest_index, "event_start_seconds"
                ] = event["start_time_seconds"]

                profile.loc[
                    nearest_index, "event_duration_seconds"
                ] = event["duration_seconds"]

                profile.loc[
                    nearest_index, "event_intensity"
                ] = event["intensity"]

                randomized_events.append(event)

            weather_profiles.append(profile)

    weather_result = pd.concat(
        weather_profiles,
        ignore_index=True
    )

    weather_result = weather_result.sort_values(
        [
            "location",
            "sol",
            "local_time_martian_hour"
        ]
    ).reset_index(drop=True)

    events_result = pd.DataFrame(
        randomized_events
    )

    if not events_result.empty:

        events_result = events_result.sort_values(
            [
                "location",
                "sol",
                "start_time_seconds"
            ]
        ).reset_index(drop=True)

    return weather_result, events_result


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(weather, events):

    expected_rows = 350

    if len(weather) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} weather records; "
            f"received {len(weather)}."
        )

    for location in weather["location"].unique():

        location_data = weather[
            weather["location"] == location
        ]

        if location_data["sol"].nunique() != 7:
            raise ValueError(
                f"{location} must have 7 randomized Sols."
            )

        for sol in sorted(
            location_data["sol"].unique()
        ):

            profile = location_data[
                location_data["sol"] == sol
            ]

            if len(profile) != 25:
                raise ValueError(
                    f"{location}, Sol {sol} "
                    "must have 25 hourly records."
                )

    if not events.empty:

        if (
            events["end_time_seconds"] > SECONDS_PER_SOL
        ).any():
            raise ValueError(
                "An event exceeds the Sol duration."
            )

        if (
            events["duration_seconds"] < MIN_EVENT_DURATION
        ).any():
            raise ValueError(
                "An event is shorter than the minimum duration."
            )

        if not np.allclose(
            events["start_time_seconds"]
            + events["duration_seconds"],
            events["end_time_seconds"],
            atol=0.02
        ):
            raise ValueError(
                "Event timing consistency check failed."
            )

        weather_ids = set(
            weather.loc[
                weather["event_id"].astype(str) != "",
                "event_id"
            ].astype(str)
        )

        event_ids = set(
            events["event_id"].astype(str)
        )

        if weather_ids != event_ids:
            raise ValueError(
                "Weather and event schedule IDs do not match."
            )

        for (location, sol), group in events.groupby(
            ["location", "sol"]
        ):

            ordered = group.sort_values(
                "start_time_seconds"
            )

            previous_end = None

            for _, event in ordered.iterrows():

                if (
                    previous_end is not None
                    and event["start_time_seconds"] < previous_end
                ):
                    raise ValueError(
                        f"Overlapping events in {location}, "
                        f"Sol {sol}."
                    )

                previous_end = event["end_time_seconds"]

    print("\nValidation passed:")
    print(f"  Weather records: {len(weather)}")
    print(f"  Events: {len(events)}")
    print("  7 Sols per location.")
    print("  25 hourly samples per Sol.")
    print("  Event IDs match between both files.")
    print("  Event durations are within their Sols.")
    print("  No event overlaps.")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Generate randomized Mars weather and events."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 65)
    print("MARS WEATHER-CONDITIONED EVENT GENERATOR V3")
    print("=" * 65)

    original_df = load_dataset(
        Path(INPUT_FILE)
    )

    print(
        f"\nLoaded {len(original_df)} original weather records."
    )

    original_df = add_game_time(original_df)

    all_events = []
    event_counter = 1

    # Generate events separately for each location.
    for location in sorted(
        original_df["location"].unique()
    ):

        print("\n" + "-" * 65)
        print(f"LOCATION: {location}")
        print("-" * 65)

        location_df = original_df[
            original_df["location"] == location
        ].copy()

        wind_threshold, median_profile = (
            calculate_wind_threshold(location_df)
        )

        location_winds = median_profile[
            "horizontal_wind_speed_m_s"
        ].to_numpy(dtype=float)

        print(
            f"\nElevated-wind threshold "
            f"(P{WIND_PERCENTILE}): "
            f"{wind_threshold:.3f} m/s"
        )

        print(
            "Original dust-storm-designated Sols:",
            sorted(
                location_df.loc[
                    location_df["dust_storm_event"] == 1,
                    "sol"
                ].unique()
            )
        )

        for source_sol in sorted(
            location_df["sol"].unique()
        ):

            sol_profile = location_df[
                location_df["sol"] == source_sol
            ].copy()

            events, event_counter = generate_sol_events(
                location=location,
                source_sol=source_sol,
                sol_profile=sol_profile,
                wind_threshold=wind_threshold,
                location_thresholds=location_winds,
                rng=rng,
                event_counter=event_counter,
            )

            all_events.extend(events)

    print("\n" + "=" * 65)
    print("SHUFFLING COMPLETE SOL PROFILES")
    print("=" * 65)

    weather_output, events_output = shuffle_complete_sols(
        original_df,
        all_events,
        rng
    )

    validate_outputs(
        weather_output,
        events_output
    )

    # Keep the original dust_storm_event column.
    # It follows the source Sol during randomization.

    weather_columns = [
        "location",
        "sol",
        "source_sol",
        "local_time_martian_hour",
        "game_time_seconds",
        "temperature_simulated_k",
        "pressure_simulated_pa",
        "solar_flux_w_m2",
        "horizontal_wind_speed_m_s",
        "daily_mean_dust_optical_depth",
        "dust_storm_event",
        "event_type",
        "event_id",
        "event_start_seconds",
        "event_duration_seconds",
        "event_intensity",
    ]

    weather_output = weather_output[
        weather_columns
    ]

    weather_output.to_csv(
        WEATHER_OUTPUT,
        index=False
    )

    events_output.to_csv(
        EVENTS_OUTPUT,
        index=False
    )

    print("\n" + "=" * 65)
    print("GENERATION COMPLETE")
    print("=" * 65)

    print(f"\nWeather file: {WEATHER_OUTPUT}")
    print(f"Events file:  {EVENTS_OUTPUT}")

    print(
        f"\nTotal weather records: {len(weather_output)}"
    )

    print(
        f"Total generated events: {len(events_output)}"
    )

    if not events_output.empty:

        print("\nEvents by location:")
        print(
            events_output["location"]
            .value_counts()
            .to_string()
        )

        print("\nEvents by type:")
        print(
            events_output["event_type"]
            .value_counts()
            .to_string()
        )

        print("\nGenerated event schedule:")
        print(
            events_output[
                [
                    "event_id",
                    "location",
                    "source_sol",
                    "sol",
                    "event_type",
                    "start_time_seconds",
                    "duration_seconds",
                    "end_time_seconds",
                ]
            ].to_string(index=False)
        )

    else:
        print(
            "\nNo events were generated. "
            "Check the wind thresholds and source labels."
        )

    print(f"\nRandom seed: {args.seed}")

    print(
        "\nReminder: MCD climatology is the weather baseline. "
        "Event selection and gameplay effects are simulation "
        "assumptions."
    )


if __name__ == "__main__":
    main()
