class FieldPresets:
    """Common desired_output_fields bundles for OP.GG MCP tools."""

    SUMMONER_PROFILE_BASIC = [
        "data.summoner.{game_name,tagline,level,profile_image_url,updated_at}",
        "data.summoner.league_stats[].{game_type,win,lose,updated_at}",
        "data.summoner.league_stats[].tier_info.{tier,division,lp,level}",
        "data.summoner.ladder_rank.{rank,total}",
        "data.summoner.recent_champion_stats[].{champion_name,play,win,kill,death,assist}",
    ]

    SUMMONER_MATCHES_BASIC = [
        "data.game_history[].{id,created_at,game_type,game_length_second,game_map}",
        "data.game_history[].average_tier_info.{tier,division,border_image_url}",
        "data.game_history[].participants[].summoner.{game_name,tagline,puuid}",
        "data.game_history[].participants[].{champion_name,items_names[],spells[],position,team_key}",
        "data.game_history[].participants[].stats.{kill,death,assist,op_score,op_score_rank,result,minion_kill,neutral_minion_kill,gold_earned,total_damage_dealt_to_champions,total_damage_taken,ward_place,vision_wards_bought_in_game}",
    ]

    MATCH_DETAIL_BASIC = [
        "data.game_detail.{id,created_at,game_type,game_length_second,game_map}",
        "data.game_detail.average_tier_info.{tier,division,border_image_url}",
        "data.game_detail.teams[].{key,banned_champions_names[]}",
        "data.game_detail.teams[].game_stat.{is_win,champion_kill,dragon_kill,baron_kill,tower_kill,gold_earned}",
        "data.game_detail.teams[].participants[].summoner.{game_name,player,puuid,tagline}",
        "data.game_detail.teams[].participants[].{champion_name,items_names[],spells[],position,team_key}",
        "data.game_detail.teams[].participants[].stats.{kill,death,assist,champion_level,op_score,op_score_rank,minion_kill,neutral_minion_kill,gold_earned,total_damage_dealt_to_champions,total_damage_taken,ward_place,vision_wards_bought_in_game,result}",
    ]

    CHAMPION_ANALYSIS_CORE = [
        "champion",
        "position",
        "data.summary.average_stats.{tier,rank,win_rate,pick_rate,ban_rate,kda,play}",
        "data.starter_items.{ids_names[],pick_rate,win}",
        "data.core_items.{ids_names[],pick_rate,win}",
        "data.boots.{ids_names[],pick_rate,win}",
        "data.summoner_spells.{ids_names[],pick_rate,win}",
        "data.runes.{primary_page_name,primary_rune_names[],secondary_page_name,secondary_rune_names[],stat_mod_names[],pick_rate,win}",
        "data.skills.{order[],pick_rate,win}",
        "data.skill_masteries.builds[].{order[],pick_rate,win}",
        "data.strong_counters[].{champion_name,play,win_rate}",
        "data.weak_counters[].{champion_name,play,win_rate}",
        "data.{damage_type,mythic_items}",
    ]

    CHAMPION_SYNERGIES = [
        "champion",
        "my_position",
        "synergy_position",
        "data.synergies[].{champion_name,synergy_champion_name,synergy_position,score,score_rank,play,win_rate}",
        "data.synergies[].synergy_tier_data.{rank,rank_prev,rank_prev_patch,tier}",
    ]

    CHAMPION_DETAILS_BASIC = [
        "requested_champions[]",
        "data.champions[].{id,key,name,title,blurb,lore,tags[],partype,release_date,ally_tips[],enemy_tips[]}",
        "data.champions[].info.{attack,defense,difficulty,magic}",
        "data.champions[].passive.{name,description}",
        "data.champions[].spells[].{key,name,description,tooltip,max_rank,cooldown_burn[],cost_burn[],range_burn[]}",
        "data.champions[].stats.{hp,hpperlevel,mp,mpperlevel,movespeed,armor,armorperlevel,spellblock,spellblockperlevel,attackdamage,attackdamageperlevel,attackspeed,attackspeedperlevel,attackrange,hpregen,hpregenperlevel,mpregen,mpregenperlevel,crit,critperlevel}",
    ]

    CHAMPION_LEADERBOARD_BASIC = [
        "champion",
        "region",
        "leaderboard[].rank",
        "leaderboard[].summoner.{game_name,tagline,level,profile_image_url}",
        "leaderboard[].summoner.league_stats[].{game_type,win,lose}",
        "leaderboard[].summoner.league_stats[].tier_info.{tier,division,lp,level}",
        "leaderboard[].most_champion_stat.{play,win,lose,kill,death,assist,op_score,minion_kill,neutral_minion_kill,damage_dealt_to_champions,damage_taken,gold_earned}",
    ]

    CHAMPION_LIST_BASIC = [
        "lang",
        "data.champions[].{champion_id,key,name,release_date}",
    ]

    LANE_META_BASIC = [
        "position_filter",
        "lang",
        "data.positions.top[].{champion,tier,rank,rank_prev,rank_prev_patch,win_rate,pick_rate,ban_rate,kda,role_rate,play,is_rip}",
        "data.positions.jungle[].{champion,tier,rank,rank_prev,rank_prev_patch,win_rate,pick_rate,ban_rate,kda,role_rate,play,is_rip}",
        "data.positions.mid[].{champion,tier,rank,rank_prev,rank_prev_patch,win_rate,pick_rate,ban_rate,kda,role_rate,play,is_rip}",
        "data.positions.adc[].{champion,tier,rank,rank_prev,rank_prev_patch,win_rate,pick_rate,ban_rate,kda,role_rate,play,is_rip}",
        "data.positions.support[].{champion,tier,rank,rank_prev,rank_prev_patch,win_rate,pick_rate,ban_rate,kda,role_rate,play,is_rip}",
    ]

    DISCOUNTED_SKINS_BASIC = [
        "lang",
        "data[].{champion_id,champion_key,champion_name,skin_id,skin_name,cost,currency,discount_rate,started_at,ended_at}",
    ]