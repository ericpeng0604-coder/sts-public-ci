#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include "game/GameContext.h"
#include "game/Shop.h"
#include "game/Game.h"

using namespace sts;
int failures=0,checks=0;
void check(bool ok,const char *name) {
    ++checks;
    if(!ok) { ++failures; if(failures<=4) std::cerr << name << "\n"; }
}
GameContext fixture(std::uint64_t seed,bool courier,bool membership) {
    GameContext gc(CharacterClass::IRONCLAD,seed,0);
    if(courier)gc.obtainRelic(RelicId::THE_COURIER);
    if(membership)gc.obtainRelic(RelicId::MEMBERSHIP_CARD);
    return gc;
}
int main() {
    for(int seed=1;seed<=64;++seed) {
        for(bool courier:{false,true})for(bool membership:{false,true}) {
            auto gc=fixture(seed,courier,membership);
            float discount=(courier?.8f:1.f)*(membership?.5f:1.f);
            auto before=gc.merchantRng.counter;
            auto p=Shop::getNewPrice(gc,150);
            check(p>=std::round(150*.95f*discount)&&p<=std::round(150*1.05f*discount),"replacement_price_range");
            check(gc.merchantRng.counter==before+1,"replacement_rng_draw_count");
        }
        for(bool colorless:{false,true}) {
            auto baseline=fixture(seed,false,false);
            auto courier=fixture(seed,true,false);
            auto member=fixture(seed,false,true);
            auto both=fixture(seed,true,true);
            auto p=Shop::getNewCardPrice(baseline,CardRarity::RARE,colorless);
            auto c=Shop::getNewCardPrice(courier,CardRarity::RARE,colorless);
            auto m=Shop::getNewCardPrice(member,CardRarity::RARE,colorless);
            auto b=Shop::getNewCardPrice(both,CardRarity::RARE,colorless);
            check(c>=p*.8f-1&&c<=p*.8f+1,"courier_only_card_discount");
            check(m>=p*.5f-1&&m<=p*.5f+1,"membership_only_card_discount");
            check(b>=p*.4f-1&&b<=p*.4f+1,"combined_card_discount");
        }
        auto gc=fixture(seed,true,false);gc.gold=10000;
        Shop shop{};shop.relics[0]=RelicId::ANCHOR;shop.relicPrice(0)=777;
        shop.buyRelic(gc,0);
        auto base=getRelicBasePrice(shop.relics[0]);
        check(shop.relicPrice(0)>=std::round(base*.95f*.8f)&&shop.relicPrice(0)<=std::round(base*1.05f*.8f),"courier_relic_slot_repriced");
        auto price=shop.relicPrice(0),gold=gc.gold;
        shop.buyRelic(gc,0);
        check(gc.gold==gold-price,"next_relic_purchase_charged_new_price");
    }
    std::cout << "{\"checks\":" << checks << ",\"failures\":" << failures << "}\n";
    return failures?1:0;
}
