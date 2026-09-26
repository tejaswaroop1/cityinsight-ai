import pandas as pd
import numpy as np


# ============================================================
# CIVICGUARD AI
# DATA ANALYTICS + INSIGHT ENGINE
# ============================================================


def numeric_series(df, column):
    """Safely convert a dataframe column to numeric."""

    if column not in df.columns:
        return pd.Series(dtype="float64")

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# SMART CITY
# ============================================================

def analyze_smart_city(df):

    data = df.copy()

    data["occupied_clean"] = (
        data["occupied"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    data["timestamp_clean"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
        utc=True
    )

    occupancy = (
        data["occupied_clean"].mean() * 100
    )

    occupied_count = int(
        data["occupied_clean"].sum()
    )

    unoccupied_count = int(
        len(data) - occupied_count
    )

    sensor_count = (
        data["sensor_id"]
        .nunique()
    )

    hourly = (
        data.dropna(
            subset=["timestamp_clean"]
        )
        .assign(
            hour=lambda x:
                x["timestamp_clean"].dt.hour
        )
        .groupby("hour")["occupied_clean"]
        .mean()
        .mul(100)
        .reset_index(
            name="Occupancy %"
        )
    )

    if not hourly.empty:

        peak_row = hourly.loc[
            hourly["Occupancy %"].idxmax()
        ]

        peak_hour = int(
            peak_row["hour"]
        )

        peak_occupancy = float(
            peak_row["Occupancy %"]
        )

    else:

        peak_hour = None
        peak_occupancy = None

    sensor_utilization = (
        data.groupby("sensor_id")[
            "occupied_clean"
        ]
        .mean()
        .mul(100)
        .reset_index(
            name="Occupancy %"
        )
        .sort_values(
            "Occupancy %",
            ascending=False
        )
    )

    return {
        "records": len(data),
        "occupancy_pct": occupancy,
        "occupied_count": occupied_count,
        "unoccupied_count": unoccupied_count,
        "sensor_count": sensor_count,
        "hourly": hourly,
        "peak_hour": peak_hour,
        "peak_occupancy": peak_occupancy,
        "sensor_utilization": sensor_utilization,
    }


# ============================================================
# FINANCE
# ============================================================

def analyze_finance(df):

    data = df.copy()

    tn_2010 = (
        numeric_series(
            data,
            "2010-11Tamil Nadu( Rs.in Crores)"
        )
    )

    india_2010 = (
        numeric_series(
            data,
            "2010-11All India( Rs.in Crores)"
        )
    )

    tn_2011 = (
        numeric_series(
            data,
            "2011-12Tamil Nadu( Rs.in Crores)"
        )
    )

    india_2011 = (
        numeric_series(
            data,
            "2011-12All India( Rs.in Crores)"
        )
    )

    result = data.copy()

    result["TN 2010-11"] = tn_2010
    result["India 2010-11"] = india_2010
    result["TN 2011-12"] = tn_2011
    result["India 2011-12"] = india_2011

    result["TN Growth"] = np.where(
        tn_2010 != 0,
        (
            (tn_2011 - tn_2010)
            / tn_2010
        ) * 100,
        np.nan
    )

    result["India Growth"] = np.where(
        india_2010 != 0,
        (
            (india_2011 - india_2010)
            / india_2010
        ) * 100,
        np.nan
    )

    return {
        "data": result,
        "regions": (
            result["Region"]
            .dropna()
            .astype(str)
            .tolist()
            if "Region" in result.columns
            else []
        ),
        "tn_2010_total": tn_2010.sum(),
        "tn_2011_total": tn_2011.sum(),
        "india_2010_total": india_2010.sum(),
        "india_2011_total": india_2011.sum(),
    }


# ============================================================
# CYBER
# ============================================================

def analyze_cyber(df):

    data = df.copy()

    data["Year"] = numeric_series(
        data,
        "Year"
    )

    data["Financial Loss"] = numeric_series(
        data,
        "Financial Loss (in Million $)"
    )

    data["Affected Users"] = numeric_series(
        data,
        "Number of Affected Users"
    )

    data["Resolution Hours"] = numeric_series(
        data,
        "Incident Resolution Time (in Hours)"
    )

    incident_count = len(data)

    total_affected = (
        data["Affected Users"]
        .fillna(0)
        .sum()
    )

    total_loss = (
        data["Financial Loss"]
        .fillna(0)
        .sum()
    )

    average_resolution = (
        data["Resolution Hours"]
        .mean()
    )

    attack_types = (
        data["Attack Type"]
        .value_counts()
        .reset_index()
    )

    attack_types.columns = [
        "Attack Type",
        "Incidents"
    ]

    industries = (
        data["Target Industry"]
        .value_counts()
        .reset_index()
    )

    industries.columns = [
        "Target Industry",
        "Incidents"
    ]

    vulnerabilities = (
        data["Security Vulnerability Type"]
        .value_counts()
        .reset_index()
    )

    vulnerabilities.columns = [
        "Vulnerability",
        "Incidents"
    ]

    sources = (
        data["Attack Source"]
        .value_counts()
        .reset_index()
    )

    sources.columns = [
        "Attack Source",
        "Incidents"
    ]

    defenses = (
        data["Defense Mechanism Used"]
        .value_counts()
        .reset_index()
    )

    defenses.columns = [
        "Defense Mechanism",
        "Incidents"
    ]

    yearly = (
        data.dropna(
            subset=["Year"]
        )
        .groupby("Year")
        .agg(
            Incidents=("Year", "size"),
            Affected_Users=(
                "Affected Users",
                "sum"
            ),
            Financial_Loss=(
                "Financial Loss",
                "sum"
            )
        )
        .reset_index()
    )

    return {
        "data": data,
        "incident_count": incident_count,
        "total_affected": total_affected,
        "total_loss": total_loss,
        "average_resolution": average_resolution,
        "attack_types": attack_types,
        "industries": industries,
        "vulnerabilities": vulnerabilities,
        "sources": sources,
        "defenses": defenses,
        "yearly": yearly,
    }


# ============================================================
# SOCIAL MEDIA
# ============================================================

def analyze_social(df):

    data = df.copy()

    numeric_columns = [
        "Daily_Minutes_Spent",
        "Posts_Per_Day",
        "Likes_Per_Day",
        "Follows_Per_Day"
    ]

    for column in numeric_columns:

        data[column] = numeric_series(
            data,
            column
        )

    average_minutes = (
        data["Daily_Minutes_Spent"]
        .mean()
    )

    average_posts = (
        data["Posts_Per_Day"]
        .mean()
    )

    average_likes = (
        data["Likes_Per_Day"]
        .mean()
    )

    average_follows = (
        data["Follows_Per_Day"]
        .mean()
    )

    app_summary = (
        data.groupby("App")
        .agg(
            Users=("User_ID", "nunique"),
            Average_Minutes=(
                "Daily_Minutes_Spent",
                "mean"
            ),
            Average_Posts=(
                "Posts_Per_Day",
                "mean"
            ),
            Average_Likes=(
                "Likes_Per_Day",
                "mean"
            ),
            Average_Follows=(
                "Follows_Per_Day",
                "mean"
            )
        )
        .reset_index()
    )

    app_summary = app_summary.sort_values(
        "Average_Minutes",
        ascending=False
    )

    return {
        "data": data,
        "users": data["User_ID"].nunique(),
        "apps": data["App"].nunique(),
        "average_minutes": average_minutes,
        "average_posts": average_posts,
        "average_likes": average_likes,
        "average_follows": average_follows,
        "app_summary": app_summary,
    }


# ============================================================
# ENVIRONMENT
# ============================================================

def analyze_environment(df):

    data = df.copy()

    numeric_columns = [
        "latitude",
        "longitude",
        "pollutant_min",
        "pollutant_max",
        "pollutant_avg"
    ]

    for column in numeric_columns:

        data[column] = numeric_series(
            data,
            column
        )

    city_summary = (
        data.groupby("city")
        .agg(
            Average_Pollutant=(
                "pollutant_avg",
                "mean"
            ),
            Minimum_Pollutant=(
                "pollutant_avg",
                "min"
            ),
            Maximum_Pollutant=(
                "pollutant_avg",
                "max"
            ),
            Stations=(
                "station",
                "nunique"
            ),
            Records=(
                "pollutant_avg",
                "count"
            )
        )
        .reset_index()
        .sort_values(
            "Average_Pollutant",
            ascending=False
        )
    )

    pollutant_summary = (
        data.groupby("pollutant_id")
        .agg(
            Average=(
                "pollutant_avg",
                "mean"
            ),
            Minimum=(
                "pollutant_avg",
                "min"
            ),
            Maximum=(
                "pollutant_avg",
                "max"
            ),
            Records=(
                "pollutant_avg",
                "count"
            )
        )
        .reset_index()
    )

    station_summary = (
        data.groupby(
            [
                "city",
                "station"
            ]
        )
        .agg(
            Average_Pollutant=(
                "pollutant_avg",
                "mean"
            ),
            Records=(
                "pollutant_avg",
                "count"
            )
        )
        .reset_index()
    )

    return {
        "data": data,
        "city_summary": city_summary,
        "pollutant_summary": pollutant_summary,
        "station_summary": station_summary,
        "average_pollution": data[
            "pollutant_avg"
        ].mean(),
        "city_count": data["city"].nunique(),
        "station_count": data["station"].nunique(),
        "pollutant_count": data[
            "pollutant_id"
        ].nunique(),
    }


# ============================================================
# DISASTER MANAGEMENT
# ============================================================

def analyze_disaster(df):

    data = df.copy()

    numeric_columns = [
        "Start Year",
        "Total Deaths",
        "No. Injured",
        "No. Affected",
        "No. Homeless",
        "Total Affected",
        "Reconstruction Costs ('000 US$)",
        "Total Damage ('000 US$)"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = numeric_series(
                data,
                column
            )

    total_events = len(data)

    total_deaths = (
        data["Total Deaths"]
        .fillna(0)
        .sum()
    )

    total_injured = (
        data["No. Injured"]
        .fillna(0)
        .sum()
    )

    total_affected = (
        data["No. Affected"]
        .fillna(0)
        .sum()
    )

    total_homeless = (
        data["No. Homeless"]
        .fillna(0)
        .sum()
    )

    total_damage = (
        data[
            "Total Damage ('000 US$)"
        ]
        .fillna(0)
        .sum()
    )

    reconstruction_cost = (
        data[
            "Reconstruction Costs ('000 US$)"
        ]
        .fillna(0)
        .sum()
    )

    yearly = (
        data.dropna(
            subset=["Start Year"]
        )
        .groupby("Start Year")
        .agg(
            Events=(
                "Start Year",
                "size"
            ),
            Deaths=(
                "Total Deaths",
                "sum"
            ),
            Injured=(
                "No. Injured",
                "sum"
            ),
            Affected=(
                "No. Affected",
                "sum"
            )
        )
        .reset_index()
    )

    disaster_types = (
        data["Disaster Type"]
        .value_counts()
        .reset_index()
    )

    disaster_types.columns = [
        "Disaster Type",
        "Events"
    ]

    disaster_subtypes = (
        data["Disaster Subtype"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    disaster_subtypes.columns = [
        "Disaster Subtype",
        "Events"
    ]

    return {
        "data": data,
        "total_events": total_events,
        "total_deaths": total_deaths,
        "total_injured": total_injured,
        "total_affected": total_affected,
        "total_homeless": total_homeless,
        "total_damage": total_damage,
        "reconstruction_cost": reconstruction_cost,
        "yearly": yearly,
        "disaster_types": disaster_types,
        "disaster_subtypes": disaster_subtypes,
    }


# ============================================================
# AUTOMATIC INSIGHT ENGINE
# ============================================================

def generate_insights(
    smart,
    finance,
    cyber,
    social,
    environment,
    disaster
):
    """
    Generate evidence-based observations.

    Every insight contains:

    Title
    Observation
    Evidence
    Implication
    Recommended Action
    """

    insights = []

    # --------------------------------------------------------
    # SMART CITY INSIGHT
    # --------------------------------------------------------

    if smart["peak_hour"] is not None:

        insights.append(
            {
                "domain": "Smart City",

                "title":
                    "Peak sensor occupancy period",

                "observation":
                    (
                        "Sensor occupancy reaches its "
                        "highest observed level around "
                        f"{smart['peak_hour']:02d}:00."
                    ),

                "evidence":
                    (
                        f"Peak recorded occupancy: "
                        f"{smart['peak_occupancy']:.1f}%."
                    ),

                "implication":
                    (
                        "This period represents the "
                        "highest observed utilization "
                        "in the sensor dataset."
                    ),

                "action":
                    (
                        "Consider prioritizing resource "
                        "monitoring around the observed "
                        "peak period."
                    )
            }
        )

    # --------------------------------------------------------
    # CYBER INSIGHT
    # --------------------------------------------------------

    if not cyber["attack_types"].empty:

        top_attack = (
            cyber["attack_types"]
            .iloc[0]
        )

        share = (
            top_attack["Incidents"]
            / cyber["incident_count"]
            * 100
            if cyber["incident_count"] > 0
            else 0
        )

        insights.append(
            {
                "domain": "Cyber Security",

                "title":
                    "Most frequently observed attack type",

                "observation":
                    (
                        f"{top_attack['Attack Type']} "
                        "is the most frequently recorded "
                        "attack category."
                    ),

                "evidence":
                    (
                        f"{int(top_attack['Incidents'])} "
                        f"of {cyber['incident_count']} "
                        f"incidents ({share:.1f}%) "
                        "belong to this category."
                    ),

                "implication":
                    (
                        "This attack category represents "
                        "a substantial portion of the "
                        "observed cyber incidents."
                    ),

                "action":
                    (
                        "Prioritize monitoring and response "
                        "planning for this observed attack "
                        "category."
                    )
            }
        )

    # --------------------------------------------------------
    # SOCIAL INSIGHT
    # --------------------------------------------------------

    if not social["app_summary"].empty:

        top_app = (
            social["app_summary"]
            .iloc[0]
        )

        insights.append(
            {
                "domain": "Social Media",

                "title":
                    "Highest observed app activity",

                "observation":
                    (
                        f"{top_app['App']} has the highest "
                        "average daily usage among the "
                        "apps in the dataset."
                    ),

                "evidence":
                    (
                        f"Average usage: "
                        f"{top_app['Average_Minutes']:.1f} "
                        "minutes per day."
                    ),

                "implication":
                    (
                        "This platform represents the "
                        "highest observed usage level "
                        "within this dataset."
                    ),

                "action":
                    (
                        "Use the observed activity pattern "
                        "when planning social-channel "
                        "communication or outreach."
                    )
            }
        )

    # --------------------------------------------------------
    # ENVIRONMENT INSIGHT
    # --------------------------------------------------------

    if not environment["city_summary"].empty:

        highest_city = (
            environment["city_summary"]
            .iloc[0]
        )

        insights.append(
            {
                "domain": "Environment",

                "title":
                    "City with highest observed pollutant average",

                "observation":
                    (
                        f"{highest_city['city']} has the "
                        "highest average pollutant value "
                        "among the cities represented."
                    ),

                "evidence":
                    (
                        f"Average pollutant value: "
                        f"{highest_city['Average_Pollutant']:.2f}."
                    ),

                "implication":
                    (
                        "The city has the highest observed "
                        "pollutant average in this dataset; "
                        "this is not an AQI measurement."
                    ),

                "action":
                    (
                        "Consider closer pollutant monitoring "
                        "at the affected locations."
                    )
            }
        )

    # --------------------------------------------------------
    # DISASTER INSIGHT
    # --------------------------------------------------------

    if not disaster["disaster_types"].empty:

        top_disaster = (
            disaster["disaster_types"]
            .iloc[0]
        )

        share = (
            top_disaster["Events"]
            / disaster["total_events"]
            * 100
            if disaster["total_events"] > 0
            else 0
        )

        insights.append(
            {
                "domain": "Disaster Management",

                "title":
                    "Most frequently recorded disaster type",

                "observation":
                    (
                        f"{top_disaster['Disaster Type']} "
                        "is the most frequently recorded "
                        "disaster type."
                    ),

                "evidence":
                    (
                        f"{int(top_disaster['Events'])} "
                        f"of {disaster['total_events']} "
                        f"events ({share:.1f}%) "
                        "belong to this type."
                    ),

                "implication":
                    (
                        "This disaster category represents "
                        "the largest share of recorded events "
                        "in the historical dataset."
                    ),

                "action":
                    (
                        "Consider incorporating this event "
                        "category into preparedness and "
                        "response planning."
                    )
            }
        )

    return insights


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_all(data):

    smart = analyze_smart_city(
        data["smart_city"]
    )

    finance = analyze_finance(
        data["finance"]
    )

    cyber = analyze_cyber(
        data["cyber"]
    )

    social = analyze_social(
        data["social"]
    )

    environment = analyze_environment(
        data["environment"]
    )

    disaster = analyze_disaster(
        data["disaster"]
    )

    insights = generate_insights(
        smart,
        finance,
        cyber,
        social,
        environment,
        disaster
    )

    return {
        "smart_city": smart,
        "finance": finance,
        "cyber": cyber,
        "social": social,
        "environment": environment,
        "disaster": disaster,
        "insights": insights,
    }