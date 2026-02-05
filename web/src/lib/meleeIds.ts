/**
 * Melee character and action IDs
 * Source: SlippiLab (MIT) - https://github.com/frankborden/slippilab
 * Original: https://github.com/project-slippi/slippi-wiki/blob/master/SPEC.md#melee-ids
 */

// External character ID -> name (used in Slippi replays)
export const characterNameByExternalId = [
  "Captain Falcon", "Donkey Kong", "Fox", "Mr. Game & Watch", "Kirby",
  "Bowser", "Link", "Luigi", "Mario", "Marth", "Mewtwo", "Ness", "Peach",
  "Pikachu", "Ice Climbers", "Jigglypuff", "Samus", "Yoshi", "Zelda", "Sheik",
  "Falco", "Young Link", "Dr. Mario", "Roy", "Pichu", "Ganondorf",
] as const;

// External character ID -> ZIP filename
export const characterZipByExternalId: Record<number, string> = {
  0: "captainFalcon", 1: "donkeyKong", 2: "fox", 3: "mrGameAndWatch", 4: "kirby",
  5: "bowser", 6: "link", 7: "luigi", 8: "mario", 9: "marth", 10: "mewtwo",
  11: "ness", 12: "peach", 13: "pikachu", 14: "iceClimbers", 15: "jigglypuff",
  16: "samus", 17: "yoshi", 18: "zelda", 19: "sheik", 20: "falco",
  21: "youngLink", 22: "doctorMario", 23: "roy", 24: "pichu", 25: "ganondorf",
};

// Internal character ID -> external character ID (for animation lookup)
export const externalIdByInternalId: Record<number, number> = {
  0: 8, 1: 2, 2: 0, 3: 1, 4: 4, 5: 5, 6: 6, 7: 19, 8: 11, 9: 12,
  10: 14, 11: 14, 12: 13, 13: 16, 14: 17, 15: 15, 16: 10, 17: 7,
  18: 9, 19: 18, 20: 21, 21: 22, 22: 20, 23: 24, 24: 3, 25: 25, 26: 23,
};

// Action state ID -> action name
export const actionNameById = [
  "DeadDown", "DeadLeft", "DeadRight", "DeadUp", "DeadUpStar", "DeadUpStarIce",
  "DeadUpFall", "DeadUpFallHitCamera", "DeadUpFallHitCameraFlat", "DeadUpFallIce",
  "DeadUpFallHitCameraIce", "Sleep", "Rebirth", "RebirthWait", "Wait", "WalkSlow",
  "WalkMiddle", "WalkFast", "Turn", "TurnRun", "Dash", "Run", "RunDirect", "RunBrake",
  "KneeBend", "JumpF", "JumpB", "JumpAerialF", "JumpAerialB", "Fall", "FallF", "FallB",
  "FallAerial", "FallAerialF", "FallAerialB", "FallSpecial", "FallSpecialF", "FallSpecialB",
  "DamageFall", "Squat", "SquatWait", "SquatRv", "Landing", "LandingFallSpecial",
  "Attack11", "Attack12", "Attack13", "Attack100Start", "Attack100Loop", "Attack100End",
  "AttackDash", "AttackS3Hi", "AttackS3HiS", "AttackS3S", "AttackS3LwS", "AttackS3Lw",
  "AttackHi3", "AttackLw3", "AttackS4Hi", "AttackS4HiS", "AttackS4S", "AttackS4LwS",
  "AttackS4Lw", "AttackHi4", "AttackLw4", "AttackAirN", "AttackAirF", "AttackAirB",
  "AttackAirHi", "AttackAirLw", "LandingAirN", "LandingAirF", "LandingAirB", "LandingAirHi",
  "LandingAirLw", "DamageHi1", "DamageHi2", "DamageHi3", "DamageN1", "DamageN2", "DamageN3",
  "DamageLw1", "DamageLw2", "DamageLw3", "DamageAir1", "DamageAir2", "DamageAir3",
  "DamageFlyHi", "DamageFlyN", "DamageFlyLw", "DamageFlyTop", "DamageFlyRoll",
  "LightGet", "HeavyGet", "LightThrowF", "LightThrowB", "LightThrowHi", "LightThrowLw",
  "LightThrowDash", "LightThrowDrop", "LightThrowAirF", "LightThrowAirB", "LightThrowAirHi",
  "LightThrowAirLw", "HeavyThrowF", "HeavyThrowB", "HeavyThrowHi", "HeavyThrowLw",
  "LightThrowF4", "LightThrowB4", "LightThrowHi4", "LightThrowLw4", "LightThrowAirF4",
  "LightThrowAirB4", "LightThrowAirHi4", "LightThrowAirLw4", "HeavyThrowF4", "HeavyThrowB4",
  "HeavyThrowHi4", "HeavyThrowLw4", "SwordSwing1", "SwordSwing3", "SwordSwing4",
  "SwordSwingDash", "BatSwing1", "BatSwing3", "BatSwing4", "BatSwingDash", "ParasolSwing1",
  "ParasolSwing3", "ParasolSwing4", "ParasolSwingDash", "HarisenSwing1", "HarisenSwing3",
  "HarisenSwing4", "HarisenSwingDash", "StarRodSwing1", "StarRodSwing3", "StarRodSwing4",
  "StarRodSwingDash", "LipStickSwing1", "LipStickSwing3", "LipStickSwing4", "LipStickSwingDash",
  "ItemParasolOpen", "ItemParasolFall", "ItemParasolFallSpecial", "ItemParasolDamageFall",
  "LGunShoot", "LGunShootAir", "LGunShootEmpty", "LGunShootAirEmpty", "FireFlowerShoot",
  "FireFlowerShootAir", "ItemScrew", "ItemScrewAir", "DamageScrew", "DamageScrewAir",
  "ItemScopeStart", "ItemScopeRapid", "ItemScopeFire", "ItemScopeEnd", "ItemScopeAirStart",
  "ItemScopeAirRapid", "ItemScopeAirFire", "ItemScopeAirEnd", "ItemScopeStartEmpty",
  "ItemScopeRapidEmpty", "ItemScopeFireEmpty", "ItemScopeEndEmpty", "ItemScopeAirStartEmpty",
  "ItemScopeAirRapidEmpty", "ItemScopeAirFireEmpty", "ItemScopeAirEndEmpty", "LiftWait",
  "LiftWalk1", "LiftWalk2", "LiftTurn", "GuardOn", "Guard", "GuardOff", "GuardSetOff",
  "GuardReflect", "DownBoundU", "DownWaitU", "DownDamageU", "DownStandU", "DownAttackU",
  "DownFowardU", "DownBackU", "DownSpotU", "DownBoundD", "DownWaitD", "DownDamageD",
  "DownStandD", "DownAttackD", "DownFowardD", "DownBackD", "DownSpotD", "Passive",
  "PassiveStandF", "PassiveStandB", "PassiveWall", "PassiveWallJump", "PassiveCeil",
  "ShieldBreakFly", "ShieldBreakFall", "ShieldBreakDownU", "ShieldBreakDownD",
  "ShieldBreakStandU", "ShieldBreakStandD", "FuraFura", "Catch", "CatchPull", "CatchDash",
  "CatchDashPull", "CatchWait", "CatchAttack", "CatchCut", "ThrowF", "ThrowB", "ThrowHi",
  "ThrowLw", "CapturePulledHi", "CaptureWaitHi", "CaptureDamageHi", "CapturePulledLw",
  "CaptureWaitLw", "CaptureDamageLw", "CaptureCut", "CaptureJump", "CaptureNeck",
  "CaptureFoot", "EscapeF", "EscapeB", "Escape", "EscapeAir", "ReboundStop", "Rebound",
  "ThrownF", "ThrownB", "ThrownHi", "ThrownLw", "ThrownLwWomen", "Pass", "Ottotto",
  "OttottoWait", "FlyReflectWall", "FlyReflectCeil", "StopWall", "StopCeil", "MissFoot",
  "CliffCatch", "CliffWait", "CliffClimbSlow", "CliffClimbQuick", "CliffAttackSlow",
  "CliffAttackQuick", "CliffEscapeSlow", "CliffEscapeQuick", "CliffJumpSlow1",
  "CliffJumpSlow2", "CliffJumpQuick1", "CliffJumpQuick2", "AppealR", "AppealL",
  "ShoulderedWait", "ShoulderedWalkSlow", "ShoulderedWalkMiddle", "ShoulderedWalkFast",
  "ShoulderedTurn", "ThrownFF", "ThrownFB", "ThrownFHi", "ThrownFLw", "CaptureCaptain",
  "CaptureYoshi", "YoshiEgg", "CaptureKoopa", "CaptureDamageKoopa", "CaptureWaitKoopa",
  "ThrownKoopaF", "ThrownKoopaB", "CaptureKoopaAir", "CaptureDamageKoopaAir",
  "CaptureWaitKoopaAir", "ThrownKoopaAirF", "ThrownKoopaAirB", "CaptureKirby",
  "CaptureWaitKirby", "ThrownKirbyStar", "ThrownCopyStar", "ThrownKirby", "BarrelWait",
  "Bury", "BuryWait", "BuryJump", "DamageSong", "DamageSongWait", "DamageSongRv",
  "DamageBind", "CaptureMewtwo", "CaptureMewtwoAir", "ThrownMewtwo", "ThrownMewtwoAir",
  "WarpStarJump", "WarpStarFall", "HammerWait", "HammerWalk", "HammerTurn", "HammerKneeBend",
  "HammerFall", "HammerJump", "HammerLanding", "KinokoGiantStart", "KinokoGiantStartAir",
  "KinokoGiantEnd", "KinokoGiantEndAir", "KinokoSmallStart", "KinokoSmallStartAir",
  "KinokoSmallEnd", "KinokoSmallEndAir", "Entry", "EntryStart", "EntryEnd", "DamageIce",
  "DamageIceJump", "CaptureMasterhand", "CapturedamageMasterhand", "CapturewaitMasterhand",
  "ThrownMasterhand", "CaptureCrazyhand", "CapturedamageCrazyhand", "CapturewaitCrazyhand",
  "ThrownCrazyhand", "CaptureKirbyYoshi", "KirbyYoshiEgg", "CaptureLeadead",
  "CaptureLikelike", "DownReflect", "CaptureCrazyhandAir", "CapturedamageCrazyhandAir",
  "CapturewaitCrazyhandAir", "ThrownCrazyhandAir", "CaptureMasterhandAir",
  "CapturedamageMasterhandAir", "CapturewaitMasterhandAir", "ThrownMasterhandAir",
] as const;

export type ActionName = typeof actionNameById[number];
